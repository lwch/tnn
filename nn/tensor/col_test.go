package tensor

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestAppendCol(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	x.requireGrad = true
	var y Tensor
	z := y.AppendCol(x)
	fmt.Println(mat.Formatted(z.Value()))
	z.Backward(Ones(z.Dims()))
	fmt.Println(mat.Formatted(x.grad.Value()))
}
