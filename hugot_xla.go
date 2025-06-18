//go:build XLA || ALL

package hugot

import (
	"fmt"

	_ "github.com/gomlx/gomlx/backends/xla"

	"github.com/knights-analytics/hugot/options"
)

func init() {
	fmt.Println("XLLLLAAAAA")
}

func NewXLASession(opts ...options.WithOption) (*Session, error) {
	return newSession("XLA", opts...)
}
