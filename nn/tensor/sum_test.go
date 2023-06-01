package tensor

import (
	"fmt"
	"testing"

	"github.com/lwch/gonum/mat32"
)

func TestSum(t *testing.T) {
	x := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	y := x.Sum()
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat32.Formatted(x.Grad().Value()))
}

func TestSumAxis(t *testing.T) {
	x := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	y := x.SumAxis(1)
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat32.Formatted(x.Grad().Value()))
}
