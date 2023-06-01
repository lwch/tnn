package tensor

import (
	"fmt"
	"testing"

	"github.com/lwch/gonum/mat32"
)

func TestSoftmaxRows(t *testing.T) {
	x := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	x.requireGrad = true
	y := x.Softmax(0)
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(New([]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, 2, 3))
	fmt.Println(mat32.Formatted(x.Grad().Value()))
}

func TestSoftmaxCols(t *testing.T) {
	x := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	x.requireGrad = true
	y := x.Softmax(1)
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(New([]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, 2, 3))
	fmt.Println(mat32.Formatted(x.Grad().Value()))
}
