package layer

import (
	"fmt"
	"testing"
	"tnn/initializer"

	"gonum.org/v1/gonum/mat"
)

func TestConv2D(t *testing.T) {
	initializer := initializer.NewNormal(1, 0)

	// m := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	// fmt.Println(mat.Formatted(m.Slice(0, 2, 0, 2)))

	input := mat.NewVecDense(9, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	layer := NewConv2D(Shape{3, 3}, Shape{2, 2}, Stride{1, 1}, initializer)
	m := layer.Forward(input)
	fmt.Println(mat.Formatted(m))
}
