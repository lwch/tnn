package tensor

import (
	"fmt"
	"testing"

	"github.com/lwch/gonum/mat32"
)

func TestVarianceAxisRows(t *testing.T) {
	x := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	x.requireGrad = true
	y := x.VarianceAxis(0, true)
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat32.Formatted(x.Grad().Value()))
}

func TestVarianceAxisCols(t *testing.T) {
	x := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	x.requireGrad = true
	y := x.VarianceAxis(1, true)
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat32.Formatted(x.Grad().Value()))
}
