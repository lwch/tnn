package tensor

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestAdd(t *testing.T) {
	x1 := New([]float64{1, 2, 3, 4}, 2, 2)
	x2 := New([]float64{5, 6, 7, 8}, 2, 2)
	y := x1.Add(x2)
	fmt.Println(mat.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat.Formatted(x1.Grad().Value()))
	fmt.Println(mat.Formatted(x2.Grad().Value()))
}

func TestAddRowVector(t *testing.T) {
	x1 := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	x2 := New([]float64{2, 2, 2}, 1, 3)
	y := x1.Add(x2)
	fmt.Println(mat.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat.Formatted(x1.Grad().Value()))
	fmt.Println(mat.Formatted(x2.Grad().Value()))
}

func TestAddColVector(t *testing.T) {
	x1 := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	x2 := New([]float64{2, 2}, 2, 1)
	y := x1.Add(x2)
	fmt.Println(mat.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat.Formatted(x1.Grad().Value()))
	fmt.Println(mat.Formatted(x2.Grad().Value()))
}
