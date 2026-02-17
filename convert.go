package claude

import (
	"encoding/json"
	"log"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

const defaultMaxTokens int64 = 8192

func buildParams(modelName string, req *model.LLMRequest) anthropic.MessageNewParams {
	log.Printf("buildParams called with %d content(s)", len(req.Contents))
	for i, c := range req.Contents {
		log.Printf("  content[%d] role=%s parts=%d", i, c.Role, len(c.Parts))
		for j, p := range c.Parts {
			log.Printf("    part[%d]: text=%q hasFuncCall=%v hasFuncResp=%v",
				j, p.Text, p.FunctionCall != nil, p.FunctionResponse != nil)
		}
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(modelName),
		MaxTokens: defaultMaxTokens,
		Messages:  contentsToMessages(req.Contents),
	}
	if req.Config == nil {
		return params
	}
	if req.Config.SystemInstruction != nil {
		if sys := systemFromContent(req.Config.SystemInstruction); len(sys) > 0 {
			params.System = sys
		}
	}
	if req.Config.MaxOutputTokens > 0 {
		params.MaxTokens = int64(req.Config.MaxOutputTokens)
	}
	if req.Config.Temperature != nil {
		params.Temperature = anthropic.Float(float64(*req.Config.Temperature))
	}
	if req.Config.TopP != nil {
		params.TopP = anthropic.Float(float64(*req.Config.TopP))
	}
	if len(req.Config.StopSequences) > 0 {
		params.StopSequences = req.Config.StopSequences
	}
	if tools := extractTools(req.Config.Tools); len(tools) > 0 {
		params.Tools = tools
		params.ToolChoice = anthropic.ToolChoiceUnionParam{
			OfAuto: &anthropic.ToolChoiceAutoParam{
				Type:                   "auto",
				DisableParallelToolUse: anthropic.Bool(true),
			},
		}
	}
	return params
}

func systemFromContent(c *genai.Content) []anthropic.TextBlockParam {
	var parts []string
	for _, p := range c.Parts {
		if p.Text != "" {
			parts = append(parts, p.Text)
		}
	}
	if len(parts) == 0 {
		return nil
	}
	return []anthropic.TextBlockParam{{Type: "text", Text: strings.Join(parts, "\n")}}
}

func contentsToMessages(contents []*genai.Content) []anthropic.MessageParam {
	var msgs []anthropic.MessageParam
	for _, c := range contents {
		if c == nil {
			continue
		}
		blocks := partsToBlocks(c.Parts)
		if len(blocks) == 0 {
			continue
		}
		if c.Role == "model" || c.Role == "assistant" {
			msgs = append(msgs, anthropic.NewAssistantMessage(blocks...))
		} else {
			msgs = append(msgs, anthropic.NewUserMessage(blocks...))
		}
	}
	return msgs
}

func partsToBlocks(parts []*genai.Part) []anthropic.ContentBlockParamUnion {
	var blocks []anthropic.ContentBlockParamUnion
	for _, p := range parts {
		switch {
		case p.Text != "":
			blocks = append(blocks, anthropic.NewTextBlock(p.Text))
		case p.FunctionCall != nil:
			blocks = append(blocks, anthropic.NewToolUseBlock(
				p.FunctionCall.ID,
				p.FunctionCall.Args,
				p.FunctionCall.Name,
			))
		case p.FunctionResponse != nil:
			log.Printf("FunctionResponse ID: %q, Name: %q", p.FunctionResponse.ID, p.FunctionResponse.Name)
			content, _ := json.Marshal(p.FunctionResponse.Response)
			blocks = append(blocks, anthropic.NewToolResultBlock(
				p.FunctionResponse.ID,
				string(content),
				false,
			))
		}
	}
	return blocks
}

func extractTools(tools []*genai.Tool) []anthropic.ToolUnionParam {
	var result []anthropic.ToolUnionParam
	for _, t := range tools {
		for _, fd := range t.FunctionDeclarations {
			schema := schemaToMap(fd.Parameters)
			inputSchema := anthropic.ToolInputSchemaParam{
				Type:       "object",
				Properties: schema["properties"],
			}
			if req, ok := schema["required"].([]string); ok {
				inputSchema.Required = req
			}
			result = append(result, anthropic.ToolUnionParam{
				OfTool: &anthropic.ToolParam{
					Name:        fd.Name,
					Description: anthropic.String(fd.Description),
					InputSchema: inputSchema,
				},
			})
		}
	}
	return result
}

func schemaToMap(s *genai.Schema) map[string]any {
	if s == nil {
		return map[string]any{"type": "object"}
	}
	m := map[string]any{"type": strings.ToLower(string(s.Type))}
	if s.Description != "" {
		m["description"] = s.Description
	}
	if len(s.Properties) > 0 {
		props := make(map[string]any, len(s.Properties))
		for k, v := range s.Properties {
			props[k] = schemaToMap(v)
		}
		m["properties"] = props
	}
	if len(s.Required) > 0 {
		m["required"] = s.Required
	}
	if len(s.Enum) > 0 {
		m["enum"] = s.Enum
	}
	if s.Items != nil {
		m["items"] = schemaToMap(s.Items)
	}
	return m
}

func messageToLLMResponse(msg *anthropic.Message) *model.LLMResponse {
	var parts []*genai.Part
	for _, block := range msg.Content {
		switch block.Type {
		case "text":
			parts = append(parts, &genai.Part{Text: block.Text})
		case "tool_use":
			tu := block.AsToolUse()
			var args map[string]any
			json.Unmarshal(tu.Input, &args)
			parts = append(parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   tu.ID,
					Name: tu.Name,
					Args: args,
				},
			})
		}
	}
	return &model.LLMResponse{
		Content:       &genai.Content{Role: "model", Parts: parts},
		FinishReason:  stopToFinish(msg.StopReason),
		UsageMetadata: usageToMetadata(msg.Usage),
	}
}

func stopToFinish(reason anthropic.StopReason) genai.FinishReason {
	switch reason {
	case anthropic.StopReasonEndTurn:
		return genai.FinishReasonStop
	case anthropic.StopReasonMaxTokens:
		return genai.FinishReasonMaxTokens
	case anthropic.StopReasonToolUse:
		return genai.FinishReasonStop
	default:
		return genai.FinishReasonOther
	}
}

func usageToMetadata(u anthropic.Usage) *genai.GenerateContentResponseUsageMetadata {
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     int32(u.InputTokens),
		CandidatesTokenCount: int32(u.OutputTokens),
		TotalTokenCount:      int32(u.InputTokens + u.OutputTokens),
	}
}
