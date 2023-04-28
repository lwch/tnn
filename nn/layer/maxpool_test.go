package layer

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestMaxPool(t *testing.T) {
	input := mat.NewDense(1, 18, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,

		10, 11, 12,
		13, 14, 15,
		16, 17, 18,
	})
	layer := NewMaxPool(Shape{3, 3}, Kernel{2, 2, 2, 2}, Stride{2, 2})
	ctx, pred := layer.Forward(input, true)
	fmt.Println(mat.Formatted(pred))
	g, _ := layer.Backward(ctx, pred)
	fmt.Println(mat.Formatted(g))
}
