//go:build XLA || ALL

package hugot

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/gomlx/types/tensors"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
)

// FEATURE EXTRACTION

func TestFeatureExtractionPipelineXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	featureExtractionPipeline(t, session)
}

func TestFeatureExtractionPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := NewXLASession(options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	featureExtractionPipeline(t, session)
}

func TestFeatureExtractionPipelineValidationXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	featureExtractionPipelineValidation(t, session)
}

// Text classification

func TestTextClassificationPipelineXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textClassificationPipeline(t, session)
}

func TestTextClassificationPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := NewXLASession(options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textClassificationPipeline(t, session)
}

func TestTextClassificationPipelineMultiXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textClassificationPipelineMulti(t, session)
}

func TestTextClassificationPipelineMultiXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := NewXLASession(options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textClassificationPipelineMulti(t, session)
}

func TestTextClassificationPipelineValidationXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textClassificationPipelineValidation(t, session)
}

// Token classification

func TestTokenClassificationPipelineXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	tokenClassificationPipeline(t, session)
}

func TestTokenClassificationPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := NewXLASession(options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	tokenClassificationPipeline(t, session)
}

func TestTokenClassificationPipelineValidationXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	tokenClassificationPipelineValidation(t, session)
}

// Zero shot

func TestZeroShotClassificationPipelineXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	zeroShotClassificationPipeline(t, session)
}

func TestZeroShotClassificationPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := NewXLASession(options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	zeroShotClassificationPipeline(t, session)
}

func TestZeroShotClassificationPipelineValidationXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	zeroShotClassificationPipelineValidation(t, session)
}

// No same name

func TestNoSameNamePipelineXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	noSameNamePipeline(t, session)
}

func TestDestroyPipelineXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	destroyPipelines(t, session)
}

// Thread safety

func TestThreadSafetyXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	threadSafety(t, session, 500)
}

func TestThreadSafetyXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := NewXLASession(options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	threadSafety(t, session, 1000)
}

// TEMP: testing how we can do inference with XLA and gemma

type ModelConfig struct {
	NumKeyValueHeads int   `json:"num_key_value_heads"`
	HeadDim          int   `json:"head_dim"`
	NumHiddenLayers  int   `json:"num_hidden_layers"`
	EosTokenID       []int `json:"eos_token_id"`
}

func CreateCache(batchSize, numLayers, numKeyValueHeads, seqLen, headDim int) []*tensors.Tensor {
	cache := make([]*tensors.Tensor, numLayers*2)

	for layer := 0; layer < numLayers; layer++ {
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

func TestGemmaInference(t *testing.T) {
	path := "/home/testuser/repositories/onnx_models"

	onnxFilename := "gemma_model.onnx"

	options := &options.Options{
		Backend: "XLA",
		GoMLXOptions: &options.GoMLXOptions{
			XLA: true,
		},
	}

	model, err := pipelineBackends.LoadModel(path, onnxFilename, options)
	if err != nil {
		panic(err)
	}

	// read config
	fileContent, err := os.ReadFile(path + "/config.json")
	if err != nil {
		fmt.Printf("Error reading file: %v\n", err)
		return
	}

	var config ModelConfig
	err = json.Unmarshal(fileContent, &config)
	if err != nil {
		fmt.Printf("Error parsing JSON: %v\n", err)
		return
	}

	// tokenize inputs
	var batch pipelineBackends.PipelineBatch
	input := []string{"what is the capital of the Netherlands?"}
	pipelineBackends.TokenizeInputs(&batch, model.Tokenizer, input)
	pipelineBackends.CreateInputTensors(&batch, model.InputsMeta, "XLA")
	// batch.Input[0].TokenIDs = []uint32{2, 105, 2364, 107, 14070, 563, 506, 5279,
	// 	529, 506, 23933, 236881, 106, 107} //hard coded python tokens
	batchSize := len(batch.Input)
	maxSeqLength := 0
	for _, i := range batch.Input {
		maxSeqLength = max(maxSeqLength, len(i.TokenIDs))
	}

	positionIDs := make([][]int64, batchSize)

	for i := range positionIDs {
		positionIDs[i] = make([]int64, maxSeqLength)
		for j := range positionIDs[i] {
			positionIDs[i][j] = int64(j + 1)
		}
	}

	inputIDs := make([][]int64, len(batch.Input))

	for i := range inputIDs {
		for _, tokenID := range batch.Input[i].TokenIDs {
			inputIDs[i] = append(inputIDs[i], int64(tokenID))
		}
	}

	inputIDsTensor := tensors.FromAnyValue(inputIDs)
	positionIDsTensor := tensors.FromAnyValue(positionIDs)

	var modelInputs []*tensors.Tensor
	modelInputs = append(modelInputs, inputIDsTensor)
	modelInputs = append(modelInputs, positionIDsTensor)

	cache := CreateCache(batchSize, config.NumHiddenLayers, config.NumKeyValueHeads, 0, config.HeadDim)

	modelInputs = append(modelInputs, cache...)

	// text generation loop
	maxNewTokens := 1024
	generatedTokens := make([][]uint32, batchSize)
	for i := range generatedTokens {
		generatedTokens[i] = make([]uint32, 0)
	}
	for step := 0; step < maxNewTokens; step++ {
		output := model.GoMLXModel.Exec.Call(modelInputs)
		logits := output[0]
		presentKeyValues := output[1:]

		logitsData := logits.Value().([][][]float32)

		nextTokenIDs := argmax(logitsData)

		terminate := true
		for i := 0; i < batchSize; i++ {
			tokenID := int64(nextTokenIDs[i][0])
			generatedTokens[i] = append(generatedTokens[i], uint32(tokenID))

			if tokenID != int64(config.EosTokenID[1]) {
				terminate = false
			}
		}

		if terminate {
			break
		}

		newInputIDs := make([][]int64, batchSize)
		for i := 0; i < batchSize; i++ {
			newInputIDs[i] = []int64{int64(nextTokenIDs[i][0])}
		}
		inputIDsTensor = tensors.FromAnyValue(newInputIDs)

		currentPositions := positionIDsTensor.Value().([][]int64)
		newPositionIDs := make([][]int64, batchSize)
		for i := 0; i < batchSize; i++ {
			lastPos := currentPositions[i][len(currentPositions[i])-1]
			newPositionIDs[i] = []int64{lastPos + 1}
		}
		positionIDsTensor = tensors.FromAnyValue(newPositionIDs)

		modelInputs = []*tensors.Tensor{inputIDsTensor, positionIDsTensor}
		modelInputs = append(modelInputs, presentKeyValues...)

	}

	// Decode
	fmt.Println("Generated tokens for each sequence:")
	fmt.Println(pipelineBackends.Decode(generatedTokens[0], model.Tokenizer, false))

}
