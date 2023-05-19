package tensor

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestExp(t *testing.T) {
	x1 := New([]float64{1, 2, 3, 4}, 2, 2)
	y := x1.Exp()
	fmt.Println(mat.Formatted(y.Forward().Value()))
	for _, grad := range y.Backward(Ones(y.Dims())) {
		fmt.Println(mat.Formatted(grad.Value()))
	}
}
