package layer

import (
	"fmt"
	"testing"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

func TestSelfAttention1(t *testing.T) {
	l := NewAttention(4, 1, 0, true)
	x := tensor.ARange(nil, 1*3*4, consts.KFloat).Reshape(1, 3, 4)
	_, score := l.Forward(x, x, x, nil, true)
	fmt.Println(score.Float32Value())
}
