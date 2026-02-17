# claude-go-adk

A Claude/Anthropic model provider for the [Google ADK for Go][]. Drop-in
replacement for `model/gemini` — implements `model.LLM` using the [Anthropic
Messages API][].

## Install

```sh
go get github.com/louislef299/claude-go-adk
```

## Usage

```go
import (
    claude "github.com/louislef299/claude-go-adk"
    "github.com/anthropics/anthropic-sdk-go/option"
    "google.golang.org/adk/agent/llmagent"
)

model := claude.NewModel("claude-sonnet-4-5-20250929")

agent, _ := llmagent.New(llmagent.Config{
    Name:        "my_agent",
    Model:       model,
    Instruction: "You are a helpful assistant.",
    Tools:       myTools,
})
```

By default, `NewModel` reads `ANTHROPIC_API_KEY` from the environment. Pass
options to override:

```go
model := claude.NewModel("claude-sonnet-4-5-20250929",
    option.WithAPIKey("sk-ant-..."),
)
```

## Supported Features

| Feature | Status |
|---|---|
| Text generation | Supported |
| Streaming | Supported |
| Tool use (function calling) | Supported |
| System instructions | Supported |
| Temperature / TopP / StopSequences | Supported |
| MaxOutputTokens | Supported (default: 8192) |

## How It Works

The package translates between ADK's `genai` types and the Anthropic API:

- `genai.Content` (role + parts) maps to Anthropic `MessageParam`
- `genai.FunctionCall` / `genai.FunctionResponse` map to `tool_use` /
  `tool_result` blocks
- `genai.FunctionDeclaration` maps to Anthropic tool definitions with JSON
  Schema normalization
- `genai.FinishReason` is derived from Anthropic's `stop_reason`
- Streaming uses `Message.Accumulate` from the Anthropic SDK, yielding partial
  text deltas and a final aggregated response

[Anthropic Messages API]: https://docs.anthropic.com/en/api/messages
[Google ADK for Go]: https://google.github.io/adk-docs/
