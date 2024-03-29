package layer

import (
	"fmt"
	"testing"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

func TestAttention(t *testing.T) {
	l := NewAttention("attn", 4, 1, 0, false)
	x := tensor.ARange(1*3*4, consts.KFloat).Reshape(1, 3, 4)
	y := l.Forward(x, x, x, nil, false, true)
	fmt.Println(y.Float32Value())
}

func TestAttentionRope(t *testing.T) {
	l := NewAttention("attn", 4, 1, 0, true)
	x := tensor.ARange(1*3*4, consts.KFloat).Reshape(1, 3, 4)
	y := l.Forward(x, x, x, nil, false, true)
	fmt.Println(y.Float32Value())
}

func TestAttentionScore(t *testing.T) {
	l := NewAttention("attn", 4, 1, 0, false)
	x := tensor.ARange(1*3*4, consts.KFloat).Reshape(1, 3, 4)
	score := l.Score(x, x, x, nil, false, true)
	fmt.Println(score.Float32Value())
}
