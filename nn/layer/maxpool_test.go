package layer

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestMaxPool(t *testing.T) {
	input := mat.NewDense(1, 9, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	output := mat.NewDense(1, 4, []float64{
		1, 0,
		0, 1,
	})
	layer := NewMaxPool(Shape{3, 3}, Shape{2, 2}, Stride{2, 2})
	pred := layer.Forward(input)
	var grad mat.Dense
	grad.Sub(output, pred)
	g := layer.Backward(&grad)
	fmt.Println(mat.Formatted(g))
}
