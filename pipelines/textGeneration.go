package pipelines

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"text/template"
	"time"

	"github.com/knights-analytics/hugot/chatTemplates"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
)

type TextGenerationPipeline struct {
	*pipelineBackends.BasePipeline
	MaxNewTokens int
	OutputName   string
	Output       pipelineBackends.InputOutputInfo
	Template     *template.Template
	EosToken     string
}

type TextGenerationOutput struct {
	TextGenerationOutputs []string
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type TemplateData struct {
	Messages            []Message `json:"messages"`
	AddGenerationPrompt bool      `json:"add_generation_prompt"`
	EosToken            string    `json:"eos_token"`
}

func (t *TextGenerationOutput) GetOutput() []any {
	out := make([]any, len(t.TextGenerationOutputs))
	for i, textOutput := range t.TextGenerationOutputs {
		out[i] = any(textOutput)
	}
	return out
}

// WithMaxTokens allows the user to define the maximum generated tokens
func WithMaxTokens(maxToken int) pipelineBackends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) error {
		pipeline.MaxNewTokens = maxToken
		return nil
	}
}

func WithGemmaTemplate() pipelineBackends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) error {
		tmpl, err := template.New("gemma").Funcs(chatTemplates.FuncMap).Parse(chatTemplates.GemmaTemplate)
		if err != nil {
			return errors.New("parsing of gemma template failed")
		}
		pipeline.Template = tmpl
		return nil
	}
}

func WithPhiTemplate() pipelineBackends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) error {
		tmpl, err := template.New("phi").Funcs(chatTemplates.FuncMap).Parse(chatTemplates.PhiTemplate)
		if err != nil {
			return errors.New("parsing of gemma template failed")
		}
		pipeline.Template = tmpl
		return nil
	}
}

// NewTextGenerationPipeline initializes a new text generation pipeline
func NewTextGenerationPipeline(config pipelineBackends.PipelineConfig[*TextGenerationPipeline], s *options.Options, model *pipelineBackends.Model) (*TextGenerationPipeline, error) {
	defaultPipeline, err := pipelineBackends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}

	pipeline := &TextGenerationPipeline{BasePipeline: defaultPipeline}
	for _, o := range config.Options {
		err = o(pipeline)
		if err != nil {
			return nil, err
		}
	}

	if pipeline.MaxNewTokens <= 0 {
		pipeline.MaxNewTokens = 1028 // Default value if not set as per Python
	}

	err = pipeline.Validate()
	if err != nil {
		return nil, err
	}
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
	var validationErrors []error
	if len(p.Model.EosTokenIDs) == 0 {
		validationErrors = append(validationErrors, errors.New("no EOS Token IDs found"))
	}
	if p.Model.NumHiddenLayers == 0 {
		validationErrors = append(validationErrors, errors.New("num hidden layers cannot be 0"))
	}
	if p.Model.NumKeyValueHeads == 0 {
		validationErrors = append(validationErrors, errors.New("num key value heads cannot be 0"))
	}
	if p.Model.HeadDim == 0 {
		validationErrors = append(validationErrors, errors.New("head dim cannot be 0"))
	}
	return errors.Join(validationErrors...)
}

// Preprocess tokenizes the input strings.
func (p *TextGenerationPipeline) Preprocess(batch *pipelineBackends.PipelineBatch, inputs []string) error {
	start := time.Now()
	p.Model.FixedCacheSize = 0
	pipelineBackends.TokenizeInputs(batch, p.Model.Tokenizer, inputs)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, uint64(time.Since(start)))

	return pipelineBackends.CreateGenerativeInputTensors(batch, p.Model, p.Runtime)
}

func (p *TextGenerationPipeline) Forward(batch *pipelineBackends.PipelineBatch) error {
	start := time.Now()

	// generation loop
	err := pipelineBackends.RunGenerativeSessionOnBatch(batch, p.BasePipeline)
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return nil
}

func (p *TextGenerationPipeline) Postprocess(batch *pipelineBackends.PipelineBatch) (*TextGenerationOutput, error) {
	outputValues := batch.OutputValues
	output := TextGenerationOutput{
		TextGenerationOutputs: make([]string, len(batch.Input)),
	}
	for i, val := range outputValues {
		tokenIDs := val.([]int64)
		convertedTokens := make([]uint32, len(tokenIDs))
		for j, tok := range tokenIDs {
			convertedTokens[j] = uint32(tok)
		}

		decodedString, err := pipelineBackends.Decode(convertedTokens, p.Model.Tokenizer, true)

		if err != nil {
			return nil, errors.New("error in decoding generated tokens")
		}
		output.TextGenerationOutputs[i] = decodedString
	}

	return &output, nil
}

func (p *TextGenerationPipeline) Run(inputs []string) (pipelineBackends.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

func executeTemplate(tmpl *template.Template, data TemplateData) (string, error) {
	var buf bytes.Buffer
	err := tmpl.Execute(&buf, data)
	if err != nil {
		return "", err
	}
	return buf.String(), nil
}

func (p *TextGenerationPipeline) RunWithTemplate(inputs [][]Message) (pipelineBackends.PipelineBatchOutput, error) {
	// apply template to messages, returning []string
	// if template is not compliled, return error
	templatedMessages := make([]string, len(inputs))

	for i, message := range inputs {
		data := TemplateData{
			Messages:            message,
			AddGenerationPrompt: true,
			EosToken:            p.EosToken,
		}
		outputStr, err := executeTemplate(p.Template, data)
		if err != nil {
			return nil, err
		}
		templatedMessages[i] = outputStr
	}
	return p.RunPipeline(templatedMessages)
}

func (p *TextGenerationPipeline) RunPipeline(inputs []string) (*TextGenerationOutput, error) {
	var runErrors []error
	batch := pipelineBackends.NewBatch()
	batch.MaxNewTokens = p.MaxNewTokens
	defer func(*pipelineBackends.PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)

	runErrors = append(runErrors, p.Preprocess(batch, inputs))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	runErrors = append(runErrors, p.Forward(batch))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	result, postErr := p.Postprocess(batch)
	runErrors = append(runErrors, postErr)
	return result, errors.Join(runErrors...)
}
