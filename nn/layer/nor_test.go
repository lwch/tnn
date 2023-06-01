package layer

import (
	"fmt"
	"testing"

	"github.com/lwch/gonum/mat32"
	"github.com/lwch/tnn/nn/tensor"
)

func TestNor(t *testing.T) {
	x := tensor.New([]float32{1, 3, 5, 2, 4, 8}, 2, 3)
	x.SetRequireGrad(true)
	layer := NewNor()
	y := layer.Forward(x, false)
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(tensor.New([]float32{0.1, 0.3, 0.5, 0.2, 0.4, 0.6}, 2, 3))
	fmt.Println(mat32.Formatted(x.Grad().Value()))
}
