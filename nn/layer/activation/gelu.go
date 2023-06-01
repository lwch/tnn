package activation

import (
	"math"

	"github.com/lwch/tnn/nn/tensor"
)

type GeLU struct {
	*base
	alpha float32
}

func NewGeLU() Activation {
	var layer GeLU
	layer.base = new("gelu")
	layer.alpha = float32(math.Sqrt(2 / math.Pi))
	return &layer
}

func (layer *GeLU) Forward(input *tensor.Tensor, _ bool) *tensor.Tensor {
	a := input.Scale(0.5)
	n := input.Pow(3).Scale(0.044715).Add(input) // (x+0.044715x^3)
	n = n.Scale(layer.alpha)
	b := n.Tanh().Add(tensor.Ones(input.Dims()))
	return a.MulElem(b)
}
