package tensor

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSoftmaxRows(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	x.requireGrad = true
	y := x.Softmax(0)
	fmt.Println(mat.Formatted(y.Value()))
	y.Backward(New([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, 2, 3))
	fmt.Println(mat.Formatted(x.Grad().Value()))
}

func TestSoftmaxCols(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	x.requireGrad = true
	y := x.Softmax(1)
	fmt.Println(mat.Formatted(y.Value()))
	y.Backward(New([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, 2, 3))
	fmt.Println(mat.Formatted(x.Grad().Value()))
}
