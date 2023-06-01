package activation

import (
	"github.com/lwch/gonum/mat32"
	"github.com/lwch/tnn/nn/tensor"
)

type ReLU struct {
	*base
}

func NewReLU() Activation {
	var layer ReLU
	layer.base = new("relu")
	return &layer
}

func (layer *ReLU) Forward(input *tensor.Tensor, _ bool) *tensor.Tensor {
	var dense mat32.Dense
	dense.Apply(func(i, j int, v float32) float32 {
		if v < 0 {
			return 0
		}
		return 1
	}, input.Value())
	mask := tensor.FromDense(&dense)
	mask.SetName(layer.Name() + ".mask")
	w := input.MulElem(mask)
	w.SetName(layer.Name() + ".output")
	return w
}
