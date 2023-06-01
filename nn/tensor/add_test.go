package tensor

import (
	"fmt"
	"testing"

	"github.com/lwch/gonum/mat32"
)

func TestAdd(t *testing.T) {
	x1 := New([]float32{1, 2, 3, 4}, 2, 2)
	x2 := New([]float32{5, 6, 7, 8}, 2, 2)
	y := x1.Add(x2)
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat32.Formatted(x1.Grad().Value()))
	fmt.Println(mat32.Formatted(x2.Grad().Value()))
}

func TestAddRowVector(t *testing.T) {
	x1 := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	x2 := New([]float32{2, 2, 2}, 1, 3)
	y := x1.Add(x2)
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat32.Formatted(x1.Grad().Value()))
	fmt.Println(mat32.Formatted(x2.Grad().Value()))
}

func TestAddColVector(t *testing.T) {
	x1 := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	x2 := New([]float32{2, 2}, 2, 1)
	y := x1.Add(x2)
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat32.Formatted(x1.Grad().Value()))
	fmt.Println(mat32.Formatted(x2.Grad().Value()))
}
