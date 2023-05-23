package activation

import (
	"fmt"
	"testing"

	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
)

func TestRelu(t *testing.T) {
	relu := NewReLU()
	input := tensor.New([]float64{
		1, -1, 0,
		0, 1, -1,
	}, 2, 3)
	output := relu.Forward(input, false)
	fmt.Println(mat.Formatted(output.Value()))
	output.Backward(tensor.Ones(output.Dims()))
}
