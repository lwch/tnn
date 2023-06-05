package activation

import (
	"github.com/lwch/tnn/nn/layer"
	"gorgonia.org/gorgonia"
)

type Sigmoid struct {
	*base
}

func NewSigmoid() layer.Layer {
	var layer Sigmoid
	layer.base = new("sigmoid")
	return &layer
}

func (layer *Sigmoid) Forward(x *gorgonia.Node) *gorgonia.Node {
	return gorgonia.Must(gorgonia.Sigmoid(x))
}
