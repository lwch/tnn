package layer

import (
	"fmt"
	"testing"

	"github.com/lwch/tnn/nn/initializer"
	"gonum.org/v1/gonum/mat"
)

func TestRnn(t *testing.T) {
	initializer := initializer.NewNumber(1)
	const times = 3
	const output = 2
	const featureSize = 2
	layer := NewRnn(times, output, initializer)
	input := mat.NewDense(4, times*featureSize, []float64{
		0, 1, 2, 3, 4, 5,
		6, 7, 8, 9, 10, 11,
		12, 13, 14, 15, 16, 17,
		18, 19, 20, 21, 22, 23,
	})
	ctx, pred := layer.Forward(input, true)
	// fmt.Println(mat.Formatted(pred))
	grad, params := layer.Backward(ctx, pred)
	fmt.Println(mat.Formatted(params.Get("b")))
	_ = grad
}
