package loss

import (
	"fmt"
	"testing"

	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
)

func TestSoftmax(t *testing.T) {
	x := tensor.New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	loss := NewSoftmax()
	y := loss.Loss(x, tensor.Ones(2, 3))
	fmt.Println(mat.Formatted(y.Value()))
	y.Backward(tensor.Ones(2, 3))
	fmt.Println(mat.Formatted(x.Grad().Value()))
}
