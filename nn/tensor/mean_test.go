package tensor

import (
	"fmt"
	"testing"

	"github.com/lwch/gonum/mat32"
)

func TestMeanAxisRows(t *testing.T) {
	x := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	x.requireGrad = true
	m := x.MeanAxis(0)
	fmt.Println(mat32.Formatted(m.Value()))
	m.Backward(Ones(m.Dims()))
	fmt.Println(mat32.Formatted(x.Grad().Value()))
}

func TestMeanAxisCols(t *testing.T) {
	x := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	x.requireGrad = true
	m := x.MeanAxis(1)
	fmt.Println(mat32.Formatted(m.Value()))
	m.Backward(Ones(m.Dims()))
	fmt.Println(mat32.Formatted(x.Grad().Value()))
}
