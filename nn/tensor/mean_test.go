package tensor

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestMeanAxisRows(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	x.requireGrad = true
	m := x.MeanAxis(0)
	fmt.Println(mat.Formatted(m.Value()))
	m.Backward(Ones(m.Dims()))
	fmt.Println(mat.Formatted(x.Grad().Value()))
}

func TestMeanAxisCols(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	x.requireGrad = true
	m := x.MeanAxis(1)
	fmt.Println(mat.Formatted(m.Value()))
	m.Backward(Ones(m.Dims()))
	fmt.Println(mat.Formatted(x.Grad().Value()))
}
