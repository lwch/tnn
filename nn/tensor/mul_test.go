package tensor

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestMul(t *testing.T) {
	x1 := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	x2 := New([]float64{7, 8, 9, 10, 11, 12}, 3, 2)
	y := x1.Mul(x2)
	fmt.Println(mat.Formatted(y.Forward().Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat.Formatted(x1.Grad().Value()))
	fmt.Println(mat.Formatted(x2.Grad().Value()))
}
