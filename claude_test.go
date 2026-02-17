package claude

import (
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/anthropics/anthropic-sdk-go/option"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func testServer(t *testing.T, file string, contentType string) *httptest.Server {
	t.Helper()
	data, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", contentType)
		w.Write(data)
	}))
}

func testModel(t *testing.T, file string, contentType string) model.LLM {
	t.Helper()
	ts := testServer(t, file, contentType)
	t.Cleanup(ts.Close)
	return NewModel("claude-sonnet-4-5-20250929",
		AnthropicOption(option.WithBaseURL(ts.URL)),
		AnthropicOption(option.WithAPIKey("test-key")),
	)
}

func TestGenerate_text(t *testing.T) {
	m := testModel(t, "testdata/text_response.json", "application/json")
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("What is the capital of France?", "user")},
	}

	var got *model.LLMResponse
	for resp, err := range m.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatal(err)
		}
		got = resp
	}

	if got == nil {
		t.Fatal("no response")
	}
	if got.Content == nil || len(got.Content.Parts) == 0 {
		t.Fatal("no content parts")
	}
	if got.Content.Parts[0].Text != "Paris" {
		t.Errorf("got text %q, want %q", got.Content.Parts[0].Text, "Paris")
	}
	if got.Content.Role != "model" {
		t.Errorf("got role %q, want %q", got.Content.Role, "model")
	}
	if got.FinishReason != genai.FinishReasonStop {
		t.Errorf("got finish reason %q, want %q", got.FinishReason, genai.FinishReasonStop)
	}
	if got.UsageMetadata == nil {
		t.Fatal("no usage metadata")
	}
	if got.UsageMetadata.PromptTokenCount != 14 {
		t.Errorf("got prompt tokens %d, want 14", got.UsageMetadata.PromptTokenCount)
	}
}

func TestGenerate_toolCall(t *testing.T) {
	m := testModel(t, "testdata/tool_call_response.json", "application/json")
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("Show me surf spots", "user")},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name:        "get_spots_of_interest",
					Description: "Returns spots of interest.",
					Parameters: &genai.Schema{
						Type: "OBJECT",
						Properties: map[string]*genai.Schema{
							"name": {Type: "STRING", Description: "Spot name or 'all'"},
						},
					},
				}},
			}},
		},
	}

	var got *model.LLMResponse
	for resp, err := range m.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatal(err)
		}
		got = resp
	}

	if got == nil || got.Content == nil {
		t.Fatal("no response")
	}
	if len(got.Content.Parts) != 2 {
		t.Fatalf("got %d parts, want 2", len(got.Content.Parts))
	}
	// First part is text
	if got.Content.Parts[0].Text == "" {
		t.Error("first part should be text")
	}
	// Second part is function call
	fc := got.Content.Parts[1].FunctionCall
	if fc == nil {
		t.Fatal("second part should be function call")
	}
	if fc.Name != "get_spots_of_interest" {
		t.Errorf("got function name %q, want %q", fc.Name, "get_spots_of_interest")
	}
	if fc.ID != "toolu_01A" {
		t.Errorf("got function id %q, want %q", fc.ID, "toolu_01A")
	}
	if fc.Args["name"] != "all" {
		t.Errorf("got args %v, want name=all", fc.Args)
	}
}

func TestGenerateStream_text(t *testing.T) {
	m := testModel(t, "testdata/stream_text.sse", "text/event-stream")
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("Say hello", "user")},
	}

	var partials []string
	var final *model.LLMResponse
	for resp, err := range m.GenerateContent(t.Context(), req, true) {
		if err != nil {
			t.Fatal(err)
		}
		if resp.Partial {
			partials = append(partials, resp.Content.Parts[0].Text)
		} else {
			final = resp
		}
	}

	if len(partials) != 2 {
		t.Fatalf("got %d partial events, want 2", len(partials))
	}
	if partials[0] != "Hello" || partials[1] != " world" {
		t.Errorf("got partials %v, want [Hello, world]", partials)
	}
	if final == nil {
		t.Fatal("no final response")
	}
	if !final.TurnComplete {
		t.Error("final response should have TurnComplete=true")
	}
	if final.Content.Parts[0].Text != "Hello world" {
		t.Errorf("got final text %q, want %q", final.Content.Parts[0].Text, "Hello world")
	}
}

func TestConvert_systemInstruction(t *testing.T) {
	content := &genai.Content{
		Parts: []*genai.Part{
			{Text: "You are a surf analyst."},
			{Text: "Rate conditions."},
		},
	}
	got := systemFromContent(content)
	if len(got) != 1 {
		t.Fatalf("got %d blocks, want 1", len(got))
	}
	want := "You are a surf analyst.\nRate conditions."
	if got[0].Text != want {
		t.Errorf("got %q, want %q", got[0].Text, want)
	}
}

func TestConvert_schemaTypes(t *testing.T) {
	schema := &genai.Schema{
		Type: "OBJECT",
		Properties: map[string]*genai.Schema{
			"name": {Type: "STRING"},
			"tags": {Type: "ARRAY", Items: &genai.Schema{Type: "STRING"}},
		},
		Required: []string{"name"},
	}
	got := schemaToMap(schema)
	if got["type"] != "object" {
		t.Errorf("got type %q, want %q", got["type"], "object")
	}
	props := got["properties"].(map[string]any)
	nameSchema := props["name"].(map[string]any)
	if nameSchema["type"] != "string" {
		t.Errorf("got name type %q, want %q", nameSchema["type"], "string")
	}
	tagsSchema := props["tags"].(map[string]any)
	if tagsSchema["type"] != "array" {
		t.Errorf("got tags type %q, want %q", tagsSchema["type"], "array")
	}
	items := tagsSchema["items"].(map[string]any)
	if items["type"] != "string" {
		t.Errorf("got items type %q, want %q", items["type"], "string")
	}
	req := got["required"].([]string)
	if len(req) != 1 || req[0] != "name" {
		t.Errorf("got required %v, want [name]", req)
	}
}

func TestConvert_toolResultInRequest(t *testing.T) {
	contents := []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "Show spots"}}},
		{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			ID: "toolu_01A", Name: "get_spots", Args: map[string]any{"name": "all"},
		}}}},
		{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			ID: "toolu_01A", Name: "get_spots", Response: map[string]any{"result": "Ocean Beach"},
		}}}},
	}
	msgs := contentsToMessages(contents, newLogger())
	if len(msgs) != 3 {
		t.Fatalf("got %d messages, want 3", len(msgs))
	}
}
