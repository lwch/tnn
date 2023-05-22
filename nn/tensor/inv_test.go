package tensor

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestInv(t *testing.T) {
	x1 := New([]float64{1, 2, 3, 4}, 2, 2)
	y := x1.Inv()
	fmt.Println(mat.Formatted(y.Forward().Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat.Formatted(x1.Grad().Value()))
}
