package pipelines

import (
	"sync/atomic"
	"time"

	util "github.com/Knights-Analytics/HuGo/utils"

	"github.com/Knights-Analytics/HuGo/utils/checks"

	"github.com/Knights-Analytics/tokenizers"
	"github.com/phuslu/log"
	ort "github.com/yalue/onnxruntime_go"
)

// basic pipeline type used for struct composition in the other pipelines
type basePipeline struct {
	ModelPath        string
	PipelineName     string
	OrtSession       *ort.DynamicAdvancedSession
	OrtOptions       *ort.SessionOptions
	Tokenizer        *tokenizers.Tokenizer
	TokenizerOptions []tokenizers.EncodeOption
	InputsMeta       []ort.InputOutputInfo
	OutputsMeta      []ort.InputOutputInfo
	hasTokenTypeIds  bool
	hasAttentionMask bool
	OutputDim        int
	TokenizerTimings *Timings
	PipelineTimings  *Timings
}

type Timings struct {
	NumCalls uint64
	TotalNS  uint64
}

type TokenizedInput struct {
	Raw               string
	Tokens            []string
	TokenIds          []uint32
	TypeIds           []uint32
	AttentionMask     []uint32
	SpecialTokensMask []uint32
	MaxAttentionIndex int
	Offsets           []tokenizers.Offset
}

type PipelineBatch struct {
	Input                []TokenizedInput
	IdsTensor            []int64
	TypeIdsTensor        []int64
	AttentionMasksTensor []int64
	MaxSequence          int
	OutputTensor         []float32
}

func (p *basePipeline) GetOutputDim() int {
	return p.OutputDim
}

func (p *basePipeline) SetSessionOptions() {
	options, optionsError := ort.NewSessionOptions()
	checks.Check(optionsError)
	checks.Check(options.SetIntraOpNumThreads(1))
	checks.Check(options.SetInterOpNumThreads(1))
	checks.Check(options.SetCpuMemArena(true))
	p.OrtOptions = options
}

// Load the ort model supporting the pipeline
func (p *basePipeline) loadModel() {

	// Initialise tokenizer
	log.Info().Msgf("Loading Tokenizer config: %s", util.PathJoinSafe(p.ModelPath, "tokenizer.json"))
	tk, err := tokenizers.FromBytes(util.ReadFileBytes(util.PathJoinSafe(p.ModelPath, "tokenizer.json")))
	checks.Check(err)

	p.SetSessionOptions()

	log.Info().Msgf("Loading model at %s/model.onnx", p.ModelPath)

	onnxBytes := util.ReadFileBytes(util.PathJoinSafe(p.ModelPath, "model.onnx"))
	inputs, outputs, err2 := ort.GetInputOutputInfoWithONNXData(onnxBytes)
	checks.Check(err2)

	p.InputsMeta = inputs
	p.OutputsMeta = outputs

	inputNames := make([]string, len(inputs))
	for i, meta := range inputs {
		inputNames[i] = meta.Name
		switch meta.Name {
		case "token_type_ids":
			p.hasTokenTypeIds = true
		case "attention_mask":
			p.hasAttentionMask = true
		}
	}
	outputNames := make([]string, len(outputs))
	for i, meta := range outputs {
		outputNames[i] = meta.Name
	}
	session, err3 := ort.NewDynamicAdvancedSessionWithONNXData(
		util.ReadFileBytes(util.PathJoinSafe(p.ModelPath, "model.onnx")),
		inputNames,
		outputNames,
		p.OrtOptions,
	)
	checks.Check(err3)

	p.OrtSession = session
	p.Tokenizer = tk
}

func (p *basePipeline) Destroy() {
	checks.Check(p.Tokenizer.Close())
	checks.Check(p.OrtSession.Destroy())
	checks.Check(p.OrtOptions.Destroy())
}

// Preprocess the input strings in the batch
func (p *basePipeline) Preprocess(inputs []string) PipelineBatch {
	start := time.Now()

	outputs := make([]TokenizedInput, len(inputs))
	maxSequence := 0
	for i, input := range inputs {

		output := p.Tokenizer.EncodeWithOptions(input,
			true,
			p.TokenizerOptions...,
		)

		maxAttentionIndex := 0
		for j, attentionMaskValue := range output.AttentionMask {
			if attentionMaskValue != 0 {
				maxAttentionIndex = j
			}
		}

		outputs[i] = TokenizedInput{
			Raw:               input,
			Tokens:            output.Tokens,
			TokenIds:          output.IDs,
			TypeIds:           output.TypeIDs,
			AttentionMask:     output.AttentionMask,
			MaxAttentionIndex: maxAttentionIndex,
			SpecialTokensMask: output.SpecialTokensMask,
			Offsets:           output.Offsets, // we need the offsets here for postprocessing later
		}
		if maxAttentionIndex > maxSequence {
			maxSequence = maxAttentionIndex
		}
	}

	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	batch := p.convertInputToTensors(outputs, maxSequence+1)
	return batch
}

func (p *basePipeline) getInputTensors(batch PipelineBatch, actualBatchSize int64, maxSequence int64) []ort.ArbitraryTensor {
	inputTensors := make([]ort.ArbitraryTensor, len(p.InputsMeta))

	for i, input := range p.InputsMeta {
		var inputTensor *ort.Tensor[int64]
		var err error

		// create the tensor for the input name
		switch input.Name {
		case "input_ids":
			inputTensor, err = ort.NewTensor(ort.NewShape(actualBatchSize, maxSequence), batch.IdsTensor)
		case "token_type_ids":
			inputTensor, err = ort.NewTensor(ort.NewShape(actualBatchSize, maxSequence), batch.TypeIdsTensor)
		case "attention_mask":
			inputTensor, err = ort.NewTensor(ort.NewShape(actualBatchSize, maxSequence), batch.AttentionMasksTensor)
		}

		checks.Check(err)
		inputTensors[i] = inputTensor
	}
	return inputTensors
}

// Forward pass of the neural network on the tokenized input
func (p *basePipeline) Forward(batch PipelineBatch) PipelineBatch {
	start := time.Now()

	actualBatchSize := int64(len(batch.Input))
	maxSequence := int64(batch.MaxSequence)
	inputTensors := p.getInputTensors(batch, actualBatchSize, maxSequence)

	outputTensor, err4 := ort.NewEmptyTensor[float32](ort.NewShape(actualBatchSize, maxSequence, int64(p.OutputDim)))
	checks.Check(err4)
	for _, tensor := range inputTensors {
		defer func(tensor ort.ArbitraryTensor) { checks.Check(tensor.Destroy()) }(tensor)
	}

	// Run Onnx model
	checks.Check(p.OrtSession.Run(inputTensors, []ort.ArbitraryTensor{outputTensor}))
	batch.OutputTensor = outputTensor.GetData()
	defer func() { checks.Check(outputTensor.Destroy()) }()

	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return batch
}

// convert tokenized input to the format required by the onnxruntime library
func (p *basePipeline) convertInputToTensors(inputs []TokenizedInput, maxSequence int) PipelineBatch {

	tensorSize := len(inputs) * maxSequence
	counter := 0

	idsTensor := make([]int64, tensorSize)
	typeIdsTensor := make([]int64, tensorSize)
	attentionMasksTensor := make([]int64, tensorSize)

	for _, input := range inputs {
		length := len(input.TokenIds)
		for j := 0; j < maxSequence; j++ {
			if j+1 <= length {
				idsTensor[counter] = int64(input.TokenIds[j])
				if p.hasTokenTypeIds {
					typeIdsTensor[counter] = int64(input.TypeIds[j])
				}
				if p.hasAttentionMask {
					attentionMasksTensor[counter] = int64(input.AttentionMask[j])
				}
			} else {
				// padding all vectors to max sequence length
				idsTensor[counter] = 0
				typeIdsTensor[counter] = 0
				attentionMasksTensor[counter] = 0
			}
			counter++
		}
	}
	return PipelineBatch{
		Input:                inputs,
		IdsTensor:            idsTensor,
		TypeIdsTensor:        typeIdsTensor,
		AttentionMasksTensor: attentionMasksTensor,
		MaxSequence:          maxSequence,
	}
}
