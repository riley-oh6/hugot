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

func argmax3D(logits [][][]float32) []int64 {
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
	padLeft := len(model.EosTokenIDs) > 0
	// 1) determine max seq length across batch
	for _, inp := range batch.Input {
		batch.MaxSequenceLength = max(batch.MaxSequenceLength, len(inp.TokenIDs))
	}
	N, L := batch.Size, batch.MaxSequenceLength
	total := N * L

	// 2) filter out cache inputs and detect whether to append cache later
	metas := make([]InputOutputInfo, 0, len(model.InputsMeta))
	hasCache := false
	for _, m := range model.InputsMeta {
		if strings.HasPrefix(m.Name, "past_key") {
			hasCache = true
		} else {
			metas = append(metas, m)
		}
	}

	// 3) prepare result containers
	inputVals := make([]ort.Value, len(metas))
	masks := make([][]bool, N)

	// 4) build each tensor
	for mi, meta := range metas {
		backing := make([]int64, total)
		idx := 0

		for bi, inp := range batch.Input {
			seqLen := len(inp.TokenIDs)
			padLen := L - seqLen
			maskRow := make([]bool, L)

			for pos := range L {
				var v int64
				switch meta.Name {
				case "input_ids":
					if padLeft {
						if pos < padLen {
							v = 0
						} else {
							v = int64(inp.TokenIDs[pos-padLen])
							maskRow[pos] = true
						}
					} else {
						if pos < seqLen {
							v = int64(inp.TokenIDs[pos])
							maskRow[pos] = true
						}
					}
				case "token_type_ids":
					// always right-pad
					if pos < seqLen {
						v = int64(inp.TypeIDs[pos])
					}
				case "attention_mask":
					if padLeft {
						if pos >= padLen {
							v = int64(inp.AttentionMask[pos-padLen])
						}
					} else {
						if pos < seqLen {
							v = int64(inp.AttentionMask[pos])
						}
					}
				case "position_ids":
					// 1-indexed positions
					v = int64(pos + 1)
				default:
					return fmt.Errorf("unrecognized input %q", meta.Name)
				}

				backing[idx] = v
				idx++
			}

			if meta.Name == "input_ids" {
				masks[bi] = maskRow
			}
		}

		// create the ONNX Runtime tensor
		t, err := ort.NewTensor(ort.NewShape(int64(N), int64(L)), backing)
		if err != nil {
			return err
		}
		inputVals[mi] = t
	}

	// 5) append KV‐cache tensors if needed
	if hasCache {
		cacheTensors, err := CreateCacheORT(
			batch.Size,
			model.NumHiddenLayers,
			model.NumKeyValueHeads,
			model.FixedCacheSize,
			model.HeadDim,
		)
		if err != nil {
			return err
		}
		inputVals = append(inputVals, cacheTensors...)
	}

	// 6) assign and prepare cleanup
	batch.InputValues = inputVals
	batch.PaddingMask = masks
	batch.DestroyInputs = func() error {
		var agg error
		for _, t := range inputVals {
			agg = errors.Join(agg, t.Destroy())
		}
		return agg
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
	actualBatchSize := int64(batch.Size)
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
			convertedOutput[i] = ReshapeOutput(v.GetData(), p.Model.OutputsMeta[i], batch.Size, batch.PaddingMask, batch.MaxSequenceLength)
		case *ort.Tensor[int64]:
			convertedOutput[i] = ReshapeOutput(v.GetData(), p.Model.OutputsMeta[i], batch.Size, batch.PaddingMask, batch.MaxSequenceLength)
		}
	}
	// store resulting tensors
	batch.OutputValues = convertedOutput

	return err
}

func GetLastTokenLogits(predictions [][][]float32) [][]float32 {
	if len(predictions) == 0 {
		return nil
	}

	batchSize := len(predictions)
	vocabSize := len(predictions[0][0]) // assuming consistent vocab size
	result := make([][]float32, batchSize)

	for i := 0; i < batchSize; i++ {
		sequence := predictions[i]
		if len(sequence) == 0 {
			result[i] = make([]float32, vocabSize) // fill with zeros if empty
			continue
		}
		last := sequence[len(sequence)-1]
		result[i] = make([]float32, vocabSize)
		copy(result[i], last)
	}

	return result
}

func Softmax2D(logits [][]float32) [][]float32 {
	batchSize := len(logits)
	if batchSize == 0 {
		return nil
	}

	vocabSize := len(logits[0])
	output := make([][]float32, batchSize)

	for i := 0; i < batchSize; i++ {
		row := logits[i]
		if len(row) != vocabSize {
			// optional: panic or error if row sizes mismatch
			continue
		}

		// 1. Find max logit for numerical stability
		maxLogit := float32(-math.MaxFloat32)
		for _, v := range row {
			if v > maxLogit {
				maxLogit = v
			}
		}

		// 2. Compute exponentials and sum
		expSum := float32(0.0)
		expVals := make([]float32, vocabSize)
		for j, v := range row {
			expVal := float32(math.Exp(float64(v - maxLogit)))
			expVals[j] = expVal
			expSum += expVal
		}

		// 3. Normalize to get probabilities
		output[i] = make([]float32, vocabSize)
		for j, expVal := range expVals {
			output[i][j] = expVal / expSum
		}
	}

	return output
}

func TopK2D(probabilities [][]float32, k int) ([][]float32, [][]int) {
	batchSize := len(probabilities)
	topProbs := make([][]float32, batchSize)
	topIndices := make([][]int, batchSize)

	for i := 0; i < batchSize; i++ {
		row := probabilities[i]
		type kv struct {
			Index int
			Value float32
		}

		// Create sortable slice
		pairs := make([]kv, len(row))
		for j, val := range row {
			pairs[j] = kv{Index: j, Value: val}
		}

		// Sort by descending value
		sort.Slice(pairs, func(a, b int) bool {
			return pairs[a].Value > pairs[b].Value
		})

		// Take top-k
		n := k
		if len(pairs) < k {
			n = len(pairs)
		}
		topProbs[i] = make([]float32, n)
		topIndices[i] = make([]int, n)
		for j := 0; j < n; j++ {
			topProbs[i][j] = pairs[j].Value
			topIndices[i][j] = pairs[j].Index
		}
	}

	return topProbs, topIndices
}

func ComputeNewScore(score float32, topProbs [][]float32, beamIndex int) float32 {
	prob := topProbs[0][beamIndex]
	logProb := float32(math.Log(float64(prob)))
	return score - logProb
}

func SelectTopBeams(allCandidates []BeamState, beamWidth int) []BeamState {
	sort.Slice(allCandidates, func(i, j int) bool {
		return allCandidates[i].logProb < allCandidates[j].logProb
	})

	if len(allCandidates) < beamWidth {
		return allCandidates
	}
	return allCandidates[:beamWidth]
}

type BeamState struct {
	tokens      []uint32    // Generated tokens so far
	logProb     float32     // Cumulative log probability
	isFinished  bool        // Whether this beam has generated an EOS token
	inputValues []ort.Value // Track all input tensors for this beam
}

func runGenerativeBeamSearch(batch *PipelineBatch, p *BasePipeline) error {
	batchSize := batch.Size
	numBeams := p.Model.BeamSearchNumBeams

	// if numBeams not set, choose default value
	if numBeams <= 0 {
		numBeams = 5
	}

	// Map input metadata for intelligent ordering
	inputMetaMap := make(map[string]int)
	for i, inputMeta := range p.Model.InputsMeta {
		inputMetaMap[inputMeta.Name] = i
	}

	// Initialize with the original batch inputs
	initialInputs := batch.InputValues.([]ort.Value)
	sequences := []BeamState{
		{
			tokens:      []uint32{}, // Start with empty generated tokens, not input tokens
			logProb:     0,
			inputValues: initialInputs, // Start with the original inputs
		},
	}

	// Track EOS token IDs for termination
	eosTokenIDs := p.Model.EosTokenIDs

iterations:
	for step := 0; step < batch.MaxNewTokens; step++ {
		var allCandidates []BeamState

		for _, seq := range sequences {
			// Run inference with this beam's current inputs
			outputTensors := make([]ort.Value, len(p.Model.OutputsMeta))
			errOnnx := p.Model.ORTModel.Session.Run(seq.inputValues, outputTensors)
			if errOnnx != nil {
				panic(errOnnx)
			}

			logits := outputTensors[0].(*ort.Tensor[float32]).GetData()
			var predictions [][][]float32
			if step == 0 {
				dimensions := p.Model.OutputsMeta[0].Dimensions.ValuesInt()
				predictions = flatDataTo3D(logits, batch.PaddingMask, batch.MaxSequenceLength, dimensions[len(dimensions)-1])
			} else {
				// after the first iteration, the shape of the logits is (batchSize, 1, vocabSize)
				predictions = flatDataTo3DGenerativeLoop(logits, 1, int64(p.Model.VocabSize)) // Note: using 1 instead of batchSize64
			}

			nextTokenLogits := GetLastTokenLogits(predictions)
			probabilities := Softmax2D(nextTokenLogits)
			topProbs, topIndices := TopK2D(probabilities, numBeams)

			for i := 0; i < numBeams; i++ {
				nextTokenID := uint32(topIndices[0][i])
				newScore := ComputeNewScore(seq.logProb, topProbs, i)

				// Only append to generated tokens, not including input
				newTokens := make([]uint32, len(seq.tokens)+1)
				copy(newTokens, seq.tokens)
				newTokens[len(seq.tokens)] = nextTokenID

				// Check if this token is EOS
				isFinished := eosTokenIDs[int64(nextTokenID)]

				// Create new input tensors for the next iteration, similar to runGenerativeORTSessionOnBatch
				newModelInputs := make([]ort.Value, len(p.Model.InputsMeta))
				for j, inputMeta := range p.Model.InputsMeta {
					switch inputMeta.Name {
					case "input_ids":
						// For beam search, we need to pass the single new token
						// Convert uint32 to int64 to match model expectations
						newTokenTensor, err := ort.NewTensor(
							ort.NewShape(1, 1), // Single beam, single token
							[]int64{int64(nextTokenID)},
						)
						if err != nil {
							return err
						}
						newModelInputs[j] = newTokenTensor

					case "position_ids":
						// Increment position ID
						positionIDs := seq.inputValues[inputMetaMap["position_ids"]].(*ort.Tensor[int64]).GetData()
						flatPositionIDs := flatDataTo2D(
							positionIDs,
							batchSize,
							len(positionIDs)/int(batchSize),
						)
						newPositionIDs := make([]int64, batchSize)
						for k, flatPositionID := range flatPositionIDs {
							newPositionIDs[k] = flatPositionID[len(flatPositionID)-1] + 1
						}
						newPositionIDsTensor, err := ort.NewTensor(
							ort.NewShape(1, 1),
							newPositionIDs,
						)
						if err != nil {
							return err
						}
						newModelInputs[j] = newPositionIDsTensor

					case "attention_mask":
						// Extend existing attention mask
						attentionMask := seq.inputValues[inputMetaMap["attention_mask"]].(*ort.Tensor[int64]).GetData()
						extendedMask := make([]int64, len(attentionMask)+1)
						copy(extendedMask, attentionMask)
						extendedMask[len(extendedMask)-1] = 1 // Add attention for new token

						newAttentionMaskTensor, err := ort.NewTensor(
							ort.NewShape(1, int64(len(extendedMask))),
							extendedMask,
						)
						if err != nil {
							return err
						}
						newModelInputs[j] = newAttentionMaskTensor

					default:
						// Handle cache inputs (past_key_values, etc.)
						if strings.HasPrefix(inputMeta.Name, "past_key") {
							cacheInputIndex := 0
							for k, meta := range p.Model.InputsMeta {
								if k < j && strings.HasPrefix(meta.Name, "past_key") {
									cacheInputIndex++
								}
							}
							newModelInputs[j] = outputTensors[1+cacheInputIndex]
						} else {
							return fmt.Errorf("unhandled input type: %s", inputMeta.Name)
						}
					}
				}

				newSeq := BeamState{
					tokens:      newTokens, // Only generated tokens
					logProb:     newScore,
					isFinished:  isFinished, // Use proper EOS token check
					inputValues: newModelInputs,
				}

				allCandidates = append(allCandidates, newSeq)
			}
		}

		sequences = SelectTopBeams(allCandidates, numBeams)

		stop := true
		for _, seq := range sequences {
			stop = stop && seq.isFinished
		}

		if stop {
			break iterations
		}
	}

	bestSequence := sequences[0].tokens
	batch.OutputValues = make([]any, batchSize)

	// Convert generated tokens to int64 (only the newly generated ones)
	beamOutput := make([]int64, len(bestSequence))
	for i := range bestSequence {
		beamOutput[i] = int64(bestSequence[i])
	}
	batch.OutputValues[0] = beamOutput
	return nil
}

func runGenerativeORTSessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {
	batchSize := batch.Size
	batchSize64 := int64(batchSize)
	generatedTokens := make([][]int64, batchSize)
	eosTokenIDs := p.Model.EosTokenIDs

	// Map input metadata for intelligent ordering
	inputMetaMap := make(map[string]int)
	for i, inputMeta := range p.Model.InputsMeta {
		inputMetaMap[inputMeta.Name] = i
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
		// this matches the python implementation where it will continue to alternate between newline and
		// EOS until the longest output sequence terminates
		// should give an array of batchSize amount of tokens
		greedyTokens := argmax3D(logitsReshaped)
		for i, greedyToken := range greedyTokens {
			if !finish[i] {
				generatedTokens[i] = append(generatedTokens[i], greedyToken)
				if eosTokenIDs[greedyToken] {
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
					greedyTokens,
				)
				if err != nil {
					return err
				}
				newModelInputs[i] = generatedTokenTensor
			case "position_ids":
				positionIDs := inputTensors[inputMetaMap["position_ids"]].(*ort.Tensor[int64]).GetData()
				flatPositionIDs := flatDataTo2D(
					positionIDs,
					batchSize,
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
					batchSize,
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

		oldInputs := batch.InputValues.([]ort.Value)
		for _, val := range oldInputs {
			val.Destroy()
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
