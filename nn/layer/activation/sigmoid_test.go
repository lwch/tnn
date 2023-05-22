package activation

import (
	"fmt"
	"testing"

	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
)

func TestSigmoid(t *testing.T) {
	sigmoid := NewSigmoid()
	input := tensor.New([]float64{
		1, -1, 0,
		0, 1, -1,
	}, 2, 3)
	output := sigmoid.Forward(input, nil, false)
	fmt.Println(mat.Formatted(output.Value()))
	output.Backward(tensor.Ones(output.Dims()))
}
