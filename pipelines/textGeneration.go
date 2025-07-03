package pipelines

import (
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"time"

	"github.com/gomlx/gomlx/types/tensors"
	jsoniter "github.com/json-iterator/go"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/util"
)

type TextGenerationPipeline struct {
	*pipelineBackends.BasePipeline
	MaxNewTokens    int
	GeneratedTokens [][]uint32
	Config          TextGenerationPipelineConfig
	OutputName      string
	Output          pipelineBackends.InputOutputInfo
}

type TextGenerationOutput struct {
	TextGenerationOutputs [][]string
}

type TextGenerationPipelineConfig struct {
	NumKeyValueHeads int   `json:"num_key_value_heads"`
	HeadDim          int   `json:"head_dim"`
	NumHiddenLayers  int   `json:"num_hidden_layers"`
	EosTokenID       []int `json:"eos_token_id"`
}

func (t *TextGenerationOutput) GetOutput() []any {
	out := make([]any, len(t.TextGenerationOutputs))
	for i, textOutput := range t.TextGenerationOutputs {
		out[i] = any(textOutput)
	}
	return out
}

// NewTextGenerationPipeline initializes a new text generation pipeline
func NewTextGenerationPipeline(config pipelineBackends.PipelineConfig[*TextGenerationPipeline], s *options.Options, model *pipelineBackends.Model) (*TextGenerationPipeline, error) {
	defaultPipeline, err := pipelineBackends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}
	defaultPipeline.IsGenerative = true

	pipeline := &TextGenerationPipeline{BasePipeline: defaultPipeline}
	for _, o := range config.Options {
		o(pipeline)
	}

	configPath := util.PathJoinSafe(model.Path, "config.json")
	pipelineInputConfig := TextGenerationPipelineConfig{}
	mapBytes, err := util.ReadFileBytes(configPath)
	if err != nil {
		return nil, err
	}
	err = jsoniter.Unmarshal(mapBytes, &pipelineInputConfig)
	if err != nil {
		return nil, err
	}

	pipeline.Config = pipelineInputConfig
	defaultPipeline.MaxTokens = pipeline.MaxNewTokens
	// defaultPipeline.Config = pipelineInputConfig
	return pipeline, nil
}

// INTERFACE IMPLEMENTATION

func (p *TextGenerationPipeline) GetMetadata() pipelineBackends.PipelineMetadata {
	return pipelineBackends.PipelineMetadata{}
}

func (p *TextGenerationPipeline) GetModel() *pipelineBackends.Model {
	return p.BasePipeline.Model
}

func (p *TextGenerationPipeline) GetStats() []string {
	return []string{
		fmt.Sprintf("Statistics for pipeline: %s", p.PipelineName),
		fmt.Sprintf("Tokenizer: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.Model.Tokenizer.TokenizerTimings.TotalNS),
			p.Model.Tokenizer.TokenizerTimings.NumCalls,
			time.Duration(float64(p.Model.Tokenizer.TokenizerTimings.TotalNS)/math.Max(1, float64(p.Model.Tokenizer.TokenizerTimings.NumCalls)))),
		fmt.Sprintf("ONNX: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.PipelineTimings.TotalNS),
			p.PipelineTimings.NumCalls,
			time.Duration(float64(p.PipelineTimings.TotalNS)/math.Max(1, float64(p.PipelineTimings.NumCalls)))),
	}
}

func (p *TextGenerationPipeline) Validate() error {
	return nil
}

func CreateCache(batchSize, numLayers, numKeyValueHeads, seqLen, headDim int) []*tensors.Tensor {
	cache := make([]*tensors.Tensor, numLayers*2)

	for layer := range numLayers {
		keyTensor := tensors.FromScalarAndDimensions(float32(0), batchSize, numKeyValueHeads, seqLen, headDim)
		cache[layer*2] = keyTensor

		valueTensor := tensors.FromScalarAndDimensions(float32(0), batchSize, numKeyValueHeads, seqLen, headDim)
		cache[layer*2+1] = valueTensor
	}
	return cache
}

func argmax(logits [][][]float32) [][]int32 {
	batchSize := len(logits)
	if batchSize == 0 {
		return nil
	}

	output := make([][]int32, batchSize)
	for i := range output {
		output[i] = make([]int32, 1)

		if len(logits[i]) == 0 {
			output[i][0] = 0
			continue
		}

		lastTokenLogits := logits[i][len(logits[i])-1]

		maxIdx := 0
		maxVal := lastTokenLogits[0]
		for j, val := range lastTokenLogits[1:] {
			if val > maxVal {
				maxVal = val
				maxIdx = j + 1
			}
		}

		output[i][0] = int32(maxIdx)
	}

	return output
}

// Preprocess tokenizes the input strings.
func (p *TextGenerationPipeline) Preprocess(batch *pipelineBackends.PipelineBatch, inputs []string) error {
	start := time.Now()
	pipelineBackends.TokenizeInputs(batch, p.Model.Tokenizer, inputs)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, uint64(time.Since(start)))
	err := pipelineBackends.CreateInputTensors(batch, p.Model.InputsMeta, p.Runtime)
	return err
}

func (p *TextGenerationPipeline) Forward(batch *pipelineBackends.PipelineBatch) error {
	start := time.Now()
	batchSize := len(batch.Input)
	maxSeqLength := 0
	for _, i := range batch.Input {
		maxSeqLength = max(maxSeqLength, len(i.TokenIDs))
	}

	// positionID initialization
	positionIDs := make([][]int64, batchSize)
	currentPositions := make([]int64, batchSize)
	for i := range positionIDs {
		positionIDs[i] = make([]int64, maxSeqLength)
		for j := range positionIDs[i] {
			pos := int64(j + 1)
			positionIDs[i][j] = pos
		}
		currentPositions[i] = int64(maxSeqLength)
	}

	// inputID initialization
	inputIDs := make([][]int64, len(batch.Input))
	for i := range inputIDs {
		for _, tokenID := range batch.Input[i].TokenIDs {
			inputIDs[i] = append(inputIDs[i], int64(tokenID))
		}
	}

	// ONNX model input initialization
	inputIDsTensor := tensors.FromAnyValue(inputIDs)
	positionIDsTensor := tensors.FromAnyValue(positionIDs)
	modelInputs := []*tensors.Tensor{inputIDsTensor, positionIDsTensor}

	// cache initialization
	cache := CreateCache(batchSize, p.Config.NumHiddenLayers, p.Config.NumKeyValueHeads, 0, p.Config.HeadDim)
	modelInputs = append(modelInputs, cache...)

	batch.InputValues = modelInputs

	// ++++++++++++++++++++++++++++++++ GENERATION LOOP ++++++++++++++++++++++++++++++++
	err := pipelineBackends.RunSessionOnBatch(batch, p.BasePipeline)
	if err != nil {
		return err
	}

	// p.GeneratedTokens = batch.OutputValues
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return nil
}

func (p *TextGenerationPipeline) Postprocess(batch *pipelineBackends.PipelineBatch) (*TextGenerationOutput, error) {
	output := TextGenerationOutput{
		TextGenerationOutputs: make([][]string, len(p.GeneratedTokens)), // Initialize with correct length
	}

	for i, seq := range p.GeneratedTokens {
		decodedString, err := pipelineBackends.Decode(seq, p.Model.Tokenizer, false)
		if err != nil {
			return nil, err
		}
		output.TextGenerationOutputs[i] = append(output.TextGenerationOutputs[i], decodedString)
	}

	return &output, nil
}

func (p *TextGenerationPipeline) Run(inputs []string) (pipelineBackends.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

func (p *TextGenerationPipeline) RunPipeline(inputs []string) (*TextGenerationOutput, error) {
	var runErrors []error
	batch := pipelineBackends.NewBatch()
	batch.MaxNewTokens = p.MaxNewTokens
	defer func(*pipelineBackends.PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)

	fmt.Println("preprocessing")
	runErrors = append(runErrors, p.Preprocess(batch, inputs))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	fmt.Println("forward pass")
	runErrors = append(runErrors, p.Forward(batch))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	fmt.Println("postprocessing")
	result, postErr := p.Postprocess(batch)
	runErrors = append(runErrors, postErr)
	return result, errors.Join(runErrors...)
}
