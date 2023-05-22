package tensor

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSub(t *testing.T) {
	x1 := New([]float64{1, 2, 3, 4}, 2, 2)
	x2 := New([]float64{5, 6, 7, 8}, 2, 2)
	y := x1.Sub(x2)
	fmt.Println(mat.Formatted(y.Forward().Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat.Formatted(x1.Grad().Value()))
	fmt.Println(mat.Formatted(x2.Grad().Value()))
}
