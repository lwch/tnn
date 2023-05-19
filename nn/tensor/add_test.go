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
	fmt.Println(mat.Formatted(y.Forward().Value()))
	for _, grad := range y.Backward(nil) {
		fmt.Println(mat.Formatted(grad.Value()))
	}
}
