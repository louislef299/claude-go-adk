package claude

import (
	"context"
	"fmt"
	"iter"
	"log"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

type claudeModel struct {
	name          string
	client        anthropic.Client
	logger        *log.Logger
	anthropicOpts []option.RequestOption
}

// NewModel returns a model.LLM backed by the Anthropic Messages API.
// By default it reads ANTHROPIC_API_KEY from the environment.
// Use AnthropicOption to pass Anthropic SDK options, and WithDebug to enable logging.
func NewModel(modelName string, opts ...Option) model.LLM {
	m := &claudeModel{
		name:   modelName,
		logger: newLogger(),
	}
	for _, o := range opts {
		o(m)
	}
	m.client = anthropic.NewClient(m.anthropicOpts...)
	return m
}

func (m *claudeModel) Name() string { return m.name }

func (m *claudeModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	params := m.buildParams(req)
	if stream {
		return m.generateStream(ctx, params)
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		msg, err := m.client.Messages.New(ctx, params)
		if err != nil {
			yield(nil, fmt.Errorf("claude: %w", err))
			return
		}
		yield(m.messageToLLMResponse(msg), nil)
	}
}

func (m *claudeModel) generateStream(ctx context.Context, params anthropic.MessageNewParams) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		stream := m.client.Messages.NewStreaming(ctx, params)
		defer stream.Close()

		var msg anthropic.Message
		for stream.Next() {
			event := stream.Current()
			if err := msg.Accumulate(event); err != nil {
				yield(nil, fmt.Errorf("claude: accumulate: %w", err))
				return
			}
			// Yield partial text deltas as they arrive.
			if delta, ok := textDelta(event); ok {
				resp := &model.LLMResponse{
					Content: &genai.Content{
						Role:  "model",
						Parts: []*genai.Part{{Text: delta}},
					},
					Partial: true,
				}
				if !yield(resp, nil) {
					return
				}
			}
		}
		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("claude: stream: %w", err))
			return
		}
		// Yield the fully accumulated response.
		resp := m.messageToLLMResponse(&msg)
		resp.TurnComplete = true
		yield(resp, nil)
	}
}

// textDelta extracts text from a content_block_delta event, if present.
func textDelta(event anthropic.MessageStreamEventUnion) (string, bool) {
	if event.Type != "content_block_delta" {
		return "", false
	}
	delta := event.AsContentBlockDelta()
	if delta.Delta.Type == "text_delta" {
		return delta.Delta.Text, true
	}
	return "", false
}
