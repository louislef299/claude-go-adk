package claude

import (
	"io"
	"log"
	"os"

	"github.com/anthropics/anthropic-sdk-go/option"
)

// Option configures the claude-go-adk adapter.
type Option func(*claudeModel)

// WithDebug enables verbose debug logging of requests and responses to
// os.Stderr.
func WithDebug() Option {
	return func(m *claudeModel) {
		m.logger = log.New(os.Stderr, "", log.LstdFlags)
	}
}

// AnthropicOption wraps an Anthropic SDK option for use with NewModel.
func AnthropicOption(opt option.RequestOption) Option {
	return func(m *claudeModel) {
		m.anthropicOpts = append(m.anthropicOpts, opt)
	}
}

func newLogger() *log.Logger {
	return log.New(io.Discard, "", 0)
}
