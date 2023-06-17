package activation

import (
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/layer"
)

type GeLU struct {
	*base
	tanh bool
}

func NewGeLU(tanh bool) *GeLU {
	var layer GeLU
	layer.base = new("gelu")
	layer.tanh = tanh
	return &layer
}

func LoadGelu(name string, _ map[string]*pb.Dense, args map[string]float32) layer.Layer {
	var layer GeLU
	layer.base = new("gelu")
	layer.name = name
	if args["tanh"] > 0 {
		layer.tanh = true
	}
	return &layer
}

func (layer *GeLU) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.Gelu(layer.tanh)
}

func (layer *GeLU) Args() map[string]float32 {
	var tanh float32
	if layer.tanh {
		tanh = 1.
	}
	return map[string]float32{
		"tanh": tanh,
	}
}
