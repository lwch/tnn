package tensor

import (
	"fmt"
	"testing"

	"github.com/lwch/gonum/mat32"
)

func TestMul(t *testing.T) {
	x1 := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	x2 := New([]float32{7, 8, 9, 10, 11, 12}, 3, 2)
	y := x1.Mul(x2)
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat32.Formatted(x1.Grad().Value()))
	fmt.Println(mat32.Formatted(x2.Grad().Value()))
}

func TestMulElem(t *testing.T) {
	x1 := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	x2 := New([]float32{7, 8, 9, 10, 11, 12}, 2, 3)
	y := x1.MulElem(x2)
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat32.Formatted(x1.Grad().Value()))
	fmt.Println(mat32.Formatted(x2.Grad().Value()))
}
