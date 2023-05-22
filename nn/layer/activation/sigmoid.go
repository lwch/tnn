package activation

import (
	"github.com/lwch/tnn/internal/math"
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
)

type Sigmoid struct {
	*base
}

func NewSigmoid() Activation {
	var layer Sigmoid
	layer.base = new("sigmoid")
	return &layer
}

func (layer *Sigmoid) Forward(input *tensor.Tensor, _ *params.List, _ bool) *tensor.Tensor {
	return math.Sigmoid(input)
}
