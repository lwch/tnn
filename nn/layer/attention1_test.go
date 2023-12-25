package layer

import (
	"fmt"
	"testing"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

func TestAttention1(t *testing.T) {
	l := NewAttention1(4, 1, 0)
	x := tensor.ARange(1*3*4, consts.KFloat).Reshape(1, 3, 4)
	y := l.Forward(x, x, x, nil, false, true)
	fmt.Println(y.Float32Value())
}

func TestAttention1Score(t *testing.T) {
	l := NewAttention1(4, 1, 0)
	x := tensor.ARange(1*3*4, consts.KFloat).Reshape(1, 3, 4)
	y := l.Score(x, x, x, nil, false, true)
	fmt.Println(y.Float32Value())
}
