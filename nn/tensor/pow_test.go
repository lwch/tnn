package tensor

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestPow(t *testing.T) {
	x := New([]float64{1, 2, 3, 4}, 2, 2)
	y := x.Pow(2)
	fmt.Println(mat.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat.Formatted(x.Grad().Value()))
}
