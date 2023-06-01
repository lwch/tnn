package loss

import (
	"fmt"
	"testing"

	"github.com/lwch/gonum/mat32"
	"github.com/lwch/tnn/nn/tensor"
)

func TestSoftmax(t *testing.T) {
	x := tensor.New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	loss := NewSoftmax()
	y := loss.Loss(x, tensor.Ones(2, 3))
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(tensor.Ones(2, 3))
	fmt.Println(mat32.Formatted(x.Grad().Value()))
}
