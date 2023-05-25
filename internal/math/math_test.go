package math

import (
	"fmt"
	"testing"

	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
)

func TestSigmoid(t *testing.T) {
	x := tensor.New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	x.SetName("x")
	y := Sigmoid(x)
	y.SetName("y")
	grad := tensor.Ones(2, 3)
	grad.SetName("grad")
	y.Backward(grad)
	fmt.Println(mat.Formatted(y.Value()))
	fmt.Println(mat.Formatted(x.Grad().Value()))
}

func TestSoftmax(t *testing.T) {
	x := tensor.New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	x.SetName("x")
	y := Softmax(x, 0)
	y.SetName("y")
	grad := tensor.Ones(2, 3)
	grad.SetName("grad")
	y.Backward(grad)
	fmt.Println(mat.Formatted(y.Value()))
	fmt.Println(mat.Formatted(x.Grad().Value()))
}
