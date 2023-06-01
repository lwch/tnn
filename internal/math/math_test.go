package math

import (
	"fmt"
	"testing"

	"github.com/lwch/gonum/mat32"
	"github.com/lwch/tnn/nn/tensor"
)

func TestSigmoid(t *testing.T) {
	x := tensor.New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	x.SetName("x")
	y := Sigmoid(x)
	y.SetName("y")
	grad := tensor.Ones(2, 3)
	grad.SetName("grad")
	y.Backward(grad)
	fmt.Println(mat32.Formatted(y.Value()))
	fmt.Println(mat32.Formatted(x.Grad().Value()))
}

// func TestSoftmax(t *testing.T) {
// 	x := tensor.New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
// 	x.SetRequireGrad(true)
// 	y := Softmax(x, 1)
// 	y.Backward(tensor.New([]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, 2, 3))
// 	fmt.Println(mat32.Formatted(y.Value()))
// 	fmt.Println(mat32.Formatted(x.Grad().Value()))
// }

func TestLogSoftmax(t *testing.T) {
	x := tensor.New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	x.SetName("x")
	y := LogSoftmax(x, 1)
	y.SetName("y")
	grad := tensor.Ones(2, 3)
	grad.SetName("grad")
	y.Backward(grad)
	fmt.Println(mat32.Formatted(y.Value()))
	fmt.Println(mat32.Formatted(x.Grad().Value()))
}
