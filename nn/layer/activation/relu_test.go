package activation

import (
	"fmt"
	"testing"

	"github.com/lwch/gonum/mat32"
	"github.com/lwch/tnn/nn/tensor"
)

func TestRelu(t *testing.T) {
	relu := NewReLU()
	input := tensor.New([]float32{
		1, -1, 0,
		0, 1, -1,
	}, 2, 3)
	output := relu.Forward(input, false)
	fmt.Println(mat32.Formatted(output.Value()))
	output.Backward(tensor.Ones(output.Dims()))
}
