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

// func TestSoftmax(t *testing.T) {
// 	x := tensor.New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
// 	x.SetRequireGrad(true)
// 	y := Softmax(x, 1)
// 	y.Backward(tensor.New([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, 2, 3))
// 	fmt.Println(mat.Formatted(y.Value()))
// 	fmt.Println(mat.Formatted(x.Grad().Value()))
// }

func TestLogSoftmax(t *testing.T) {
	x := tensor.New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	x.SetName("x")
	y := LogSoftmax(x, 1)
	y.SetName("y")
	grad := tensor.Ones(2, 3)
	grad.SetName("grad")
	y.Backward(grad)
	fmt.Println(mat.Formatted(y.Value()))
	fmt.Println(mat.Formatted(x.Grad().Value()))
}
