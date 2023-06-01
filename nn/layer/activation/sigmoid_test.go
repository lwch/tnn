package activation

import (
	"fmt"
	"testing"

	"github.com/lwch/gonum/mat32"
	"github.com/lwch/tnn/nn/tensor"
)

func TestSigmoid(t *testing.T) {
	sigmoid := NewSigmoid()
	input := tensor.New([]float32{
		1, -1, 0,
		0, 1, -1,
	}, 2, 3)
	output := sigmoid.Forward(input, false)
	fmt.Println(mat32.Formatted(output.Value()))
	output.Backward(tensor.Ones(output.Dims()))
}
