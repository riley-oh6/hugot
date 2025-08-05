//go:build ORT || ALL

package pipelineBackends

import (
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/knights-analytics/hugot/options"
)

type ORTModel struct {
	Session        *ort.DynamicAdvancedSession
	SessionOptions *ort.SessionOptions
	Options        *options.OrtOptions
	Destroy        func() error
}

func createORTModelBackend(model *Model, options *options.Options) error {

	// TODO: currently models with external data can only load from regular filesystems
	cwd, err := os.Getwd()
	if err != nil {
		return err
	}
	pathChanged := false
	if !strings.HasPrefix(model.Path, "s3:") {
		err = os.Chdir(model.Path)
		if err != nil {
			return err
		}
		pathChanged = true
	}

	sessionOptions := options.BackendOptions.(*ort.SessionOptions)

	inputs, outputs, err := loadInputOutputMetaORT(model.OnnxBytes)
	if err != nil {
		return err
	}

	var inputNames []string
	var outputNames []string
	for _, v := range inputs {
		inputNames = append(inputNames, v.Name)
	}
	for _, v := range outputs {
		outputNames = append(outputNames, v.Name)
	}
	session, errSession := ort.NewDynamicAdvancedSessionWithONNXData(
		model.OnnxBytes,
		inputNames,
		outputNames,
		sessionOptions,
	)
	if errSession != nil {
		return errSession
	}

	model.ORTModel = &ORTModel{
		Session:        session,
		SessionOptions: sessionOptions,
		Options:        options.ORTOptions,
		Destroy: func() error {
			return session.Destroy()
		},
	}
	model.InputsMeta = inputs
	model.OutputsMeta = outputs
	if pathChanged {
		err = os.Chdir(cwd)
	}

	return err
}

func loadInputOutputMetaORT(onnxBytes []byte) ([]InputOutputInfo, []InputOutputInfo, error) {
	inputs, outputs, err := ort.GetInputOutputInfoWithONNXData(onnxBytes)
	if err != nil {
		return nil, nil, err
	}
	return convertORTInputOutputs(inputs), convertORTInputOutputs(outputs), nil
}

func createInputTensorsORT(batch *PipelineBatch, model *Model) error {
	batchSize := len(batch.Input)
	tensorSize := batchSize * batch.MaxSequenceLength

	inputTensors := make([]ort.Value, len(model.InputsMeta))
	var tensorCreationErr error

	paddingMasks := make([][]bool, batchSize)

	for i, inputMeta := range model.InputsMeta {
		backingSlice := make([]int64, tensorSize)
		counter := 0

		for j, input := range batch.Input {
			inputPaddingMask := make([]bool, batch.MaxSequenceLength)
			length := len(input.TokenIDs)
			for k := 0; k < batch.MaxSequenceLength; k++ {
				if k+1 <= length {
					switch inputMeta.Name {
					case "input_ids":
						backingSlice[counter] = int64(input.TokenIDs[k])
						inputPaddingMask[k] = true
					case "token_type_ids":
						backingSlice[counter] = int64(input.TypeIDs[k])
					case "attention_mask":
						backingSlice[counter] = int64(input.AttentionMask[k])
					default:
						return fmt.Errorf("input %s not recognized", inputMeta.Name)
					}
				} else {
					backingSlice[counter] = 0 // pad with zero
				}
				counter++
			}

			if inputMeta.Name == "input_ids" {
				paddingMasks[j] = inputPaddingMask
			}
		}
		inputTensors[i], tensorCreationErr = ort.NewTensor(ort.NewShape(int64(batchSize), int64(batch.MaxSequenceLength)), backingSlice)
		if tensorCreationErr != nil {
			return tensorCreationErr
		}
	}
	batch.InputValues = inputTensors
	batch.PaddingMask = paddingMasks
	batch.DestroyInputs = func() error {
		var destroyError error
		for _, ortTensor := range inputTensors {
			destroyError = errors.Join(destroyError, ortTensor.Destroy())
		}
		return destroyError
	}

	return nil
}

func CreateGenerativeInputTensorsORT(batch *PipelineBatch, model *Model) error {
	for _, i := range batch.Input {
		batch.MaxSequenceLength = max(batch.MaxSequenceLength, len(i.TokenIDs))
	}

	batchSize := len(batch.Input)
	maxSeqLength := batch.MaxSequenceLength
	tensorSize := batchSize * maxSeqLength
	hasCache := false

	// filter out cache inputs
	filteredInputsMeta := make([]InputOutputInfo, 0, len(model.InputsMeta))
	for _, meta := range model.InputsMeta {
		if strings.HasPrefix(meta.Name, "past_key") {
			hasCache = true
		} else {
			filteredInputsMeta = append(filteredInputsMeta, meta)
		}
	}

	inputTensors := make([]ort.Value, len(filteredInputsMeta))
	var tensorCreationErr error
	paddingMasks := make([][]bool, batchSize)

	for i, inputMeta := range filteredInputsMeta {
		backingSlice := make([]int64, tensorSize)
		counter := 0

		for j, input := range batch.Input {
			seq := input.TokenIDs
			seqLen := len(seq)
			padLen := maxSeqLength - seqLen
			inputPaddingMask := make([]bool, maxSeqLength)

			for k := range maxSeqLength {
				switch inputMeta.Name {
				case "input_ids":
					if k < padLen {
						backingSlice[counter] = 0 // padding
						inputPaddingMask[k] = false
					} else {
						backingSlice[counter] = int64(seq[k-padLen])
						inputPaddingMask[k] = true
					}
				case "position_ids":
					backingSlice[counter] = int64(k + 1)
				case "attention_mask":
					if k < padLen {
						backingSlice[counter] = 0 // padding
					} else {
						backingSlice[counter] = 1 // attention
					}
				default:
					return fmt.Errorf("input %s not recognized", inputMeta.Name)
				}
				counter++
			}

			if inputMeta.Name == "input_ids" {
				paddingMasks[j] = inputPaddingMask
			}
		}

		inputTensors[i], tensorCreationErr = ort.NewTensor(ort.NewShape(int64(batchSize), int64(maxSeqLength)), backingSlice)
		if tensorCreationErr != nil {
			return tensorCreationErr
		}
	}

	if hasCache {
		cache, err := CreateCacheORT(batchSize, model.NumHiddenLayers, model.NumKeyValueHeads, model.FixedCacheSize, model.HeadDim)
		if err != nil {
			return err
		}
		inputTensors = append(inputTensors, cache...)
	}

	batch.InputValues = inputTensors
	batch.PaddingMask = paddingMasks
	batch.DestroyInputs = func() error {
		var destroyError error
		for _, ortTensor := range inputTensors {
			destroyError = errors.Join(destroyError, ortTensor.Destroy())
		}
		return destroyError
	}

	return nil
}

// CreateCacheORT initializes the KV cache. Cache entries have shape [batchSize, numKeyValueHeads, pastSequenceLength, headDim]
func CreateCacheORT(batchSize, numLayers, numKeyValueHeads, maxSeqLen, headDim int) ([]ort.Value, error) {
	cache := make([]ort.Value, numLayers*2)
	tensorSize := batchSize * numKeyValueHeads * maxSeqLen * headDim
	for layer := range numLayers {
		keySlice := make([]float32, tensorSize)
		keyTensor, err := ort.NewTensor(
			ort.NewShape(int64(batchSize), int64(numKeyValueHeads), int64(maxSeqLen), int64(headDim)),
			keySlice,
		)
		if err != nil {
			for i := 0; i < layer*2; i++ {
				if cache[i] != nil {
					errors.Join(err, cache[i].Destroy())
				}
			}
			return nil, err
		}
		cache[layer*2] = keyTensor
		valueSlice := make([]float32, tensorSize)
		valueTensor, err := ort.NewTensor(
			ort.NewShape(int64(batchSize), int64(numKeyValueHeads), int64(maxSeqLen), int64(headDim)),
			valueSlice,
		)
		if err != nil {
			errors.Join(err, keyTensor.Destroy())
			for i := 0; i < layer*2; i++ {
				if cache[i] != nil {
					errors.Join(err, cache[i].Destroy())
				}
			}
			return nil, err
		}
		cache[layer*2+1] = valueTensor
	}
	return cache, nil
}

func runORTSessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {
	actualBatchSize := int64(len(batch.Input))
	maxSequenceLength := int64(batch.MaxSequenceLength)
	var err error

	// allocate vectors with right dimensions for the output
	outputTensors := make([]ort.Value, len(p.Model.OutputsMeta))
	defer func() {
		for _, output := range outputTensors {
			err = errors.Join(err, output.Destroy())
		}
	}()

	for outputIndex, meta := range p.Model.OutputsMeta {
		var batchDimSet bool
		var tokenDimSet bool
		actualDims := make([]int64, 0, len(meta.Dimensions))

		for _, dim := range meta.Dimensions {
			if dim == -1 {
				if !batchDimSet {
					actualDims = append(actualDims, actualBatchSize)
					batchDimSet = true
				} else if !tokenDimSet {
					actualDims = append(actualDims, maxSequenceLength)
					tokenDimSet = true
				} else {
					return fmt.Errorf("only two axis can be dynamic (batch size and number of tokens)")
				}
			} else {
				actualDims = append(actualDims, dim)
			}
		}
		outputShape := ort.NewShape(actualDims...)
		outputTensors[outputIndex], err = ort.NewEmptyTensor[float32](outputShape)
		if err != nil {
			return err
		}
	}

	errOnnx := p.Model.ORTModel.Session.Run(batch.InputValues.([]ort.Value), outputTensors)
	if errOnnx != nil {
		return errOnnx
	}

	convertedOutput := make([]any, len(outputTensors))
	for i, t := range outputTensors {
		switch v := t.(type) {
		case *ort.Tensor[float32]:
			convertedOutput[i] = ReshapeOutput(v.GetData(), p.Model.OutputsMeta[i], batch.PaddingMask, batch.MaxSequenceLength)
		case *ort.Tensor[int64]:
			convertedOutput[i] = ReshapeOutput(v.GetData(), p.Model.OutputsMeta[i], batch.PaddingMask, batch.MaxSequenceLength)
		}
	}

	// store resulting tensors
	batch.OutputValues = convertedOutput

	return err
}

// BeamHypotheses stores and manages beam search hypotheses
type BeamHypotheses struct {
	numBeams      int
	maxLength     int
	beams         []BeamHypothesis
	lengthPenalty float32
	earlyStopping bool
}

// BeamHypothesis represents a single beam hypothesis
type BeamHypothesis struct {
	tokens []int64
	score  float32
}

// NewBeamHypotheses creates a new BeamHypotheses instance
func NewBeamHypotheses(numBeams int, maxLength int, lengthPenalty float32, earlyStopping bool) *BeamHypotheses {
	return &BeamHypotheses{
		numBeams:      numBeams,
		maxLength:     maxLength,
		beams:         make([]BeamHypothesis, 0, numBeams),
		lengthPenalty: lengthPenalty,
		earlyStopping: earlyStopping,
	}
}

// Add adds a new hypothesis to the beam
func (bh *BeamHypotheses) Add(hypothesis []int64, sumLogProbs float32, generatedLen int) {
	// Apply length penalty: (5 + len(hyp))^length_penalty / (5 + 1)^length_penalty
	// This matches the Python implementation
	score := sumLogProbs / float32(math.Pow(float64(5+generatedLen), float64(bh.lengthPenalty))) *
		float32(math.Pow(float64(6), float64(bh.lengthPenalty)))

	// Create a copy of the tokens to avoid reference issues
	tokensCopy := make([]int64, len(hypothesis))
	copy(tokensCopy, hypothesis)

	// Create the beam hypothesis
	beam := BeamHypothesis{
		tokens: tokensCopy,
		score:  score,
	}

	// Add to beams
	bh.beams = append(bh.beams, beam)

	// Sort beams by score (descending)
	sort.Slice(bh.beams, func(i, j int) bool {
		return bh.beams[i].score > bh.beams[j].score
	})

	// Keep only the top numBeams
	if len(bh.beams) > bh.numBeams {
		bh.beams = bh.beams[:bh.numBeams]
	}
}

// IsDone checks if the beam search is done
func (bh *BeamHypotheses) IsDone(bestSumLogProbs float32, curLen int) bool {
	// If we don't have enough beams yet, we're not done
	if len(bh.beams) < bh.numBeams {
		return false
	}

	if bh.earlyStopping {
		return true
	}

	// Calculate worst score in the beam
	worstScore := bh.beams[len(bh.beams)-1].score

	// Calculate the score of the best possible beam from the current state
	// Apply the same length penalty as in Add
	nextLen := curLen + 1
	bestPossibleScore := bestSumLogProbs / float32(math.Pow(float64(5+nextLen), float64(bh.lengthPenalty))) *
		float32(math.Pow(float64(6), float64(bh.lengthPenalty)))

	// If the best possible score is worse than our worst beam, we're done
	return worstScore >= bestPossibleScore
}

// GetBeams returns the final beams
func (bh *BeamHypotheses) GetBeams() []BeamHypothesis {
	return bh.beams
}

func argmax(logits [][][]float32) []int64 {
	batchSize := len(logits)
	if batchSize == 0 {
		return nil
	}

	output := make([]int64, batchSize)
	for i := range output {
		if len(logits[i]) == 0 {
			output[i] = 0
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

		output[i] = int64(maxIdx)
	}

	return output
}

// beamSearch implements beam search decoding
// This is a simplified version of beam search that selects the most likely token at each step
// For a full implementation, we would need to track multiple beams across iterations
// and select the best complete sequence at the end
// To use beam search, set the BeamSearchEnabled option to true and optionally set BeamSearchNumBeams
// Example:
//
//	opts := options.Defaults()
//	opts.Backend = "ORT"
//	err := options.WithBeamSearch(true)(opts)
//	err = options.WithBeamSearchNumBeams(5)(opts)
func beamSearch(logits [][][]float32, numBeams int) []int64 {
	batchSize := len(logits)
	if batchSize == 0 {
		return nil
	}

	// Define a struct to hold token scores
	type tokenScore struct {
		token int64
		score float32
	}

	// Create a slice to hold the next tokens for each item in the batch
	nextTokens := make([]int64, batchSize)

	// Process each item in the batch
	for batchIdx := range logits {
		// Get the logits for the last token
		if len(logits[batchIdx]) == 0 {
			nextTokens[batchIdx] = 0
			continue
		}

		lastTokenLogits := logits[batchIdx][len(logits[batchIdx])-1]
		vocabSize := len(lastTokenLogits)

		// Create a slice to hold all token scores
		allScores := make([]tokenScore, vocabSize)
		for i, score := range lastTokenLogits {
			allScores[i] = tokenScore{token: int64(i), score: score}
		}

		// Sort by score (descending)
		sort.Slice(allScores, func(i, j int) bool {
			return allScores[i].score > allScores[j].score
		})

		// Take the top numBeams (or fewer if we don't have enough tokens)
		beamCount := numBeams
		if beamCount > len(allScores) {
			beamCount = len(allScores)
		}

		topBeams := allScores[:beamCount]

		// For now, just use the highest scoring token
		// In a more complete implementation, we would track multiple beams across iterations
		// and select the best complete sequence at the end
		if len(topBeams) > 0 {
			nextTokens[batchIdx] = topBeams[0].token
		} else {
			nextTokens[batchIdx] = 0 // Default to 0 if no tokens available
		}
	}

	return nextTokens
}

// runGenerativeORTSessionOnBatch runs the generative loop for text generation
func runGenerativeORTSessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {
	batchSize := len(batch.Input)
	batchSize64 := int64(batchSize)
	generatedTokens := make([][]int64, batchSize)
	eosTokenIDs := p.Model.EosTokenIDs

	// Map input metadata for intelligent ordering
	inputMetaMap := make(map[string]int)
	for i, inputMeta := range p.Model.InputsMeta {
		inputMetaMap[inputMeta.Name] = i
	}

	// Check if beam search is enabled
	useBeamSearch := false
	numBeams := 5 // Default number of beams

	// Get beam search configuration from options if available
	if p.Model.ORTModel.Options != nil {
		if p.Model.ORTModel.Options.BeamSearchEnabled != nil {
			useBeamSearch = *p.Model.ORTModel.Options.BeamSearchEnabled
			if useBeamSearch && p.Model.ORTModel.Options.BeamSearchNumBeams != nil {
				numBeams = *p.Model.ORTModel.Options.BeamSearchNumBeams
			}
		}
	}

	// Initialize beam hypotheses if using beam search
	var beamHypotheses []*BeamHypotheses
	if useBeamSearch {
		beamHypotheses = make([]*BeamHypotheses, batchSize)
		for i := 0; i < batchSize; i++ {
			beamHypotheses[i] = NewBeamHypotheses(numBeams, batch.MaxNewTokens, 1.0, false)
		}
	}

	finish := make([]bool, batchSize)
	finishCount := 0
iterations:
	for step := 0; step < batch.MaxNewTokens; step++ {
		inputTensors := batch.InputValues.([]ort.Value)
		outputTensors := make([]ort.Value, len(p.Model.OutputsMeta))
		errOnnx := p.Model.ORTModel.Session.Run(inputTensors, outputTensors)
		if errOnnx != nil {
			return errOnnx
		}

		logits := outputTensors[0].(*ort.Tensor[float32]).GetData()
		var logitsReshaped [][][]float32
		if step == 0 {
			dimensions := p.Model.OutputsMeta[0].Dimensions.ValuesInt()
			logitsReshaped = flatDataTo3D(logits, batch.PaddingMask, batch.MaxSequenceLength, dimensions[len(dimensions)-1])
		} else {
			// after the first iteration, the shape of the logits is (batchSize, 1, vocabSize) so this is handled differently
			logitsReshaped = flatDataTo3DGenerativeLoop(logits, batchSize64, int64(p.Model.VocabSize))
		}

		var nextTokens []int64
		if useBeamSearch {
			// Use beam search for token selection
			nextTokens = beamSearch(logitsReshaped, numBeams)
		} else {
			// Use greedy search (argmax) for token selection
			nextTokens = argmax(logitsReshaped)
		}

		for i, nextToken := range nextTokens {
			if !finish[i] {
				generatedTokens[i] = append(generatedTokens[i], nextToken)
				if eosTokenIDs[nextToken] {
					finish[i] = true
					finishCount++
				}
			}
		}
		if finishCount == batchSize {
			break iterations
		}

		// initialize next loop in correct order of the input metadata
		newModelInputs := make([]ort.Value, len(p.Model.InputsMeta))
		for i, inputMeta := range p.Model.InputsMeta {
			switch inputMeta.Name {
			case "input_ids":
				generatedTokenTensor, err := ort.NewTensor(
					ort.NewShape(batchSize64, 1),
					nextTokens,
				)
				if err != nil {
					return err
				}
				newModelInputs[i] = generatedTokenTensor

			case "position_ids":
				positionIDs := inputTensors[inputMetaMap["position_ids"]].(*ort.Tensor[int64]).GetData()
				flatPositionIDs := flatDataTo2D(
					positionIDs,
					batch.PaddingMask,
					len(positionIDs)/batchSize,
				)
				newPositionIDs := make([]int64, batchSize)
				for j, flatPositionID := range flatPositionIDs {
					newPositionIDs[j] = flatPositionID[len(flatPositionID)-1] + 1
				}
				newPositionIDsTensor, err := ort.NewTensor(
					ort.NewShape(batchSize64, 1),
					newPositionIDs,
				)
				if err != nil {
					return err
				}
				newModelInputs[i] = newPositionIDsTensor

			case "attention_mask":
				attentionMask := inputTensors[inputMetaMap["attention_mask"]].(*ort.Tensor[int64]).GetData()
				flatAttentionMask := flatDataTo2D(
					attentionMask,
					batch.PaddingMask,
					len(attentionMask)/batchSize,
				)
				newAttentionMask := make([]int64, batchSize64*int64(len(flatAttentionMask[0])+1))
				counter := 0
				for j := range flatAttentionMask {
					for k := range flatAttentionMask[j] {
						newAttentionMask[counter] = flatAttentionMask[j][k]
						counter++
					}
					newAttentionMask[counter] = 1
					counter++
				}
				newAttentionMaskTensor, err := ort.NewTensor(
					ort.NewShape(batchSize64, int64(len(flatAttentionMask[0])+1)),
					newAttentionMask,
				)
				if err != nil {
					return err
				}
				newModelInputs[i] = newAttentionMaskTensor

			default:
				// handle cache inputs (past_key_values, etc.)
				if strings.HasPrefix(inputMeta.Name, "past_key") {
					cacheInputIndex := 0
					for j, meta := range p.Model.InputsMeta {
						if j < i && strings.HasPrefix(meta.Name, "past_key") {
							cacheInputIndex++
						}
					}
					newModelInputs[i] = outputTensors[1+cacheInputIndex]
				} else {
					return fmt.Errorf("unhandled input type: %s", inputMeta.Name)
				}
			}
		}

		batch.InputValues = newModelInputs
	}

	batch.OutputValues = make([]any, batchSize)
	for i := range generatedTokens {
		batch.OutputValues[i] = generatedTokens[i]
	}
	return nil
}

func convertORTInputOutputs(inputOutputs []ort.InputOutputInfo) []InputOutputInfo {
	inputOutputsStandardised := make([]InputOutputInfo, len(inputOutputs))
	for i, inputOutput := range inputOutputs {
		inputOutputsStandardised[i] = InputOutputInfo{
			Name:       inputOutput.Name,
			Dimensions: Shape(inputOutput.Dimensions),
		}
	}
	return inputOutputsStandardised
}
