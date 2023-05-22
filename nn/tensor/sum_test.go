package tensor

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSum(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	y := x.Sum()
	fmt.Println(mat.Formatted(y.Forward().Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat.Formatted(x.Grad().Value()))
}
