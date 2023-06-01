package tensor

import (
	"fmt"
	"testing"

	"github.com/lwch/gonum/mat32"
)

func TestConact(t *testing.T) {
	x1 := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	x2 := New([]float32{7, 8, 9, 10, 11, 12}, 2, 3)
	x1.requireGrad = true
	x2.requireGrad = true
	y := x1.Conact(x2)
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat32.Formatted(x1.grad.Value()))
	fmt.Println(mat32.Formatted(x2.grad.Value()))
}

func TestStack(t *testing.T) {
	x1 := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	x2 := New([]float32{7, 8, 9, 10, 11, 12}, 2, 3)
	x1.requireGrad = true
	x2.requireGrad = true
	y := x1.Stack(x2)
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat32.Formatted(x1.grad.Value()))
	fmt.Println(mat32.Formatted(x2.grad.Value()))
}
