package layer

import (
	"fmt"
	"testing"

	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
)

func TestNor(t *testing.T) {
	x := tensor.New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	x.SetRequireGrad(true)
	layer := NewNor()
	y := layer.Forward(x, false)
	fmt.Println(mat.Formatted(y.Value()))
	y.Backward(tensor.New([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, 2, 3))
	fmt.Println(mat.Formatted(x.Grad().Value()))
}
