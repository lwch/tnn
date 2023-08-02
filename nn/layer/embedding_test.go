package layer

import (
	"fmt"
	"testing"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

func TestEmbedding(t *testing.T) {
	l := NewEmbedding(5, 16)
	x := tensor.ARange(nil, 5, consts.KInt64)
	fmt.Println(l.Forward(x).Float32Value())
}
