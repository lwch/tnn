package layer

import (
	"fmt"
	"testing"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

func TestSelfAttention(t *testing.T) {
	l := NewAttention(4, 1, 0)
	x := tensor.ARange(nil, 1*3*4, consts.KFloat).Reshape(1, 3, 4)
	_, score := l.Forward(x, x, x, nil, false, true)
	fmt.Println(score.Float32Value())
}
