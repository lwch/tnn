package layer

import (
	"fmt"
	"testing"

	"github.com/lwch/tnn/initializer"
	"gonum.org/v1/gonum/mat"
)

func TestConv2D(t *testing.T) {
	initializer := initializer.NewNormal(1, 0.5)

	input := mat.NewDense(1, 9, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	output := mat.NewDense(1, 18, []float64{
		// layer1
		0, 0, 1,
		0, 1, 0,
		0, 1, 1,
		// layer2
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	})
	layer := NewConv2D(Shape{3, 3}, Kernel{2, 2, 1, 2}, Stride{1, 1}, initializer)
	pred := layer.Forward(input)
	var grad mat.Dense
	grad.Sub(output, pred)
	g := layer.Backward(&grad)
	fmt.Println(mat.Formatted(g))
}
