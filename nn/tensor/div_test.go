package tensor

import (
	"fmt"
	"testing"

	"github.com/lwch/gonum/mat32"
)

func TestDivElem(t *testing.T) {
	x1 := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	x2 := New([]float32{4, 5, 6, 7, 8, 9}, 2, 3)
	y := x1.DivElem(x2)
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat32.Formatted(x1.Grad().Value()))
	fmt.Println(mat32.Formatted(x2.Grad().Value()))
}

func TestDivElemRowVector(t *testing.T) {
	x1 := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	x2 := New([]float32{2, 2, 2}, 1, 3)
	y := x1.DivElem(x2)
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat32.Formatted(x1.Grad().Value()))
	fmt.Println(mat32.Formatted(x2.Grad().Value()))
}

func TestDivElemColVector(t *testing.T) {
	x1 := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	x2 := Ones(2, 1)
	y := x1.Sub(x2)
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat32.Formatted(x1.Grad().Value()))
	fmt.Println(mat32.Formatted(x2.Grad().Value()))
}
