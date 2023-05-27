package tensor

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestMaxAxis(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	y := x.MaxAxis(1)
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat.Formatted(y.Value()))
	fmt.Println(mat.Formatted(x.Grad().Value()))
}
