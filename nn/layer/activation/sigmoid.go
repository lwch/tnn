package activation

import (
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/layer"
)

type Sigmoid struct {
	*base
}

func NewSigmoid() *Sigmoid {
	var layer Sigmoid
	layer.base = new("sigmoid")
	return &layer
}

func LoadSigmoid(name string, _ map[string]*pb.Dense, _ map[string]float32) layer.Layer {
	var layer Sigmoid
	layer.base = new("sigmoid")
	layer.name = name
	return &layer
}

func (layer *Sigmoid) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.Sigmoid()
}
