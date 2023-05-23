package tensor

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSlice(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3)
	y := x.Slice(0, 2, 0, 2)
	fmt.Println(mat.Formatted(y.Forward().Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat.Formatted(x.Grad().Value()))
}
