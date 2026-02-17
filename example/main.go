package main

import (
	"context"
	"fmt"
	"log"

	"github.com/louislef299/claude-go-adk"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func main() {
	llm := claude.NewModel("claude-sonnet-4-5-20250929")
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What is the capital of France? One word.", "user"),
		},
		Config: &genai.GenerateContentConfig{},
	}
	for resp, err := range llm.GenerateContent(context.Background(), req, false) {
		if err != nil {
			log.Fatal(err)
		}
		for _, p := range resp.Content.Parts {
			fmt.Print(p.Text)
		}
	}
	fmt.Println()
}
