package tensor

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestVarianceAxisRows(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	x.requireGrad = true
	y := x.VarianceAxis(0)
	fmt.Println(mat.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat.Formatted(x.Grad().Value()))
}

func TestVarianceAxisCols(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	x.requireGrad = true
	y := x.VarianceAxis(1)
	fmt.Println(mat.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat.Formatted(x.Grad().Value()))
}
