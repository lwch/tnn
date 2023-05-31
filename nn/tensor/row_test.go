package tensor

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestRow2Matrix(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	x.requireGrad = true
	y := x.Row2Matrix(0, 3, 1)
	fmt.Println(mat.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat.Formatted(x.grad.Value()))
}

func TestRowVector(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	x.requireGrad = true
	y := x.RowVector()
	fmt.Println(mat.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat.Formatted(x.grad.Value()))
}

func TestAppendRow(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	x.requireGrad = true
	var y Tensor
	z := y.AppendRow(x)
	fmt.Println(mat.Formatted(z.Value()))
	z.Backward(Ones(z.Dims()))
	fmt.Println(mat.Formatted(x.grad.Value()))
}
