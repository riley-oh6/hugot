//go:build XLA || ALL

package hugot

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
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

func CreateCache(batchSize, numLayers, numKeyValueHeads, seqLen, headDim int) map[string]*tensors.Tensor {
	cache := make(map[string]*tensors.Tensor)

	for layer := 0; layer < numLayers; layer++ {
		// Create key tensor (even when seqLen == 0)
		keyShape := shapes.Make(dtypes.Float32, batchSize, numKeyValueHeads, seqLen, headDim)
		keyTensor := tensors.FromShape(keyShape)
		keyName := fmt.Sprintf("past_key_values.%d.key", layer)
		cache[keyName] = keyTensor

		// Create value tensor (same shape as key)
		valueShape := shapes.Make(dtypes.Float32, batchSize, numKeyValueHeads, seqLen, headDim)
		valueTensor := tensors.FromShape(valueShape)
		valueName := fmt.Sprintf("past_key_values.%d.value", layer)
		cache[valueName] = valueTensor
	}
	return cache
}

func TestGemmaInferenceXLA(t *testing.T) {
	// load model
	path := "/home/rpinosio/repositories/knights/gemma"
	onnxFilename := "model.onnx"

	options := &options.Options{
		Backend: "XLA",
		GoMLXOptions: &options.GoMLXOptions{
			XLA: true,
		},
	}

	model, err := pipelineBackends.LoadModel(path, onnxFilename, options)
	if err != nil {
		t.Fatal(err)
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
	err = pipelineBackends.CreateInputTensors(&batch, model.InputsMeta, "XLA")
	if err != nil {
		t.Fatal(err)
	}

	batchSize := len(batch.Input)
	inputs := batch.InputValues.([]*tensors.Tensor)
	fmt.Println("Inputs:", inputs)

	positionIDs := make([]uint32, batchSize*batch.MaxSequenceLength)
	for i := 0; i < batchSize; i++ {
		row := make([]uint32, batch.MaxSequenceLength)
		for j := 0; j < batch.MaxSequenceLength; j++ {
			row[j] = uint32(j + 1)
		}
		positionIDs = append(positionIDs, row...)
	}

	// positionIDsTensor := tensors.FromFlatDataAndDimensions(positionIDs, batchSize, batch.MaxSequenceLength)

	// batch.InputValues

	// inputs = append(inputs, batch.InputTensors...)

	// model.GoMLXModel.Exec.Call(inputs...)

	// use 1 because gomlx tensor creation fails with 0
	// cache := CreateCache(batchSize, config.NumHiddenLayers, config.NumKeyValueHeads, 1, config.HeadDim)

	// // Add all KV cache entries
	// for layer := 0; layer < config.NumHiddenLayers; layer++ {
	// 	key := fmt.Sprintf("past_key_values.%d.key", layer)
	// 	value := fmt.Sprintf("past_key_values.%d.value", layer)

	// 	// Create a map for this layer's key-value pair
	// 	kMap := map[string]*tensors.Tensor{
	// 		key: cache[key],
	// 	}
	// 	inputs = append(inputs, kMap)

	// 	vMap := map[string]*tensors.Tensor{
	// 		value: cache[value],
	// 	}
	// 	inputs = append(inputs, vMap)
	// }
	// type KeyValueTensor struct {
	//     Key   string
	//     Value *tensors.Tensor
	// }

	// for layer := 0; layer < config.NumHiddenLayers; layer++ {
	//     key := fmt.Sprintf("past_key_values.%d.key", layer)
	//     value := fmt.Sprintf("past_key_values.%d.value", layer)

	//     // Using struct instead of map
	//     kStruct := KeyValueTensor{
	//         Key:   key,
	//         Value: cache[key],
	//     }
	//     inputs = append(inputs, kStruct)

	//     vStruct := KeyValueTensor{
	//         Key:   value,
	//         Value: cache[value],
	//     }
	//     inputs = append(inputs, vStruct)
	// }

	// fmt.Println(model.GoMLXModel.OnnxModel.InputsNames)
	// fmt.Println(model.GoMLXModel.OnnxModel.InputsShapes)
	// fmt.Println("")
	// // fmt.Println(inputs)

	// fmt.Println(inputs...)

}
