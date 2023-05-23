package net

import (
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
)

type Net struct {
	layers []layer.Layer
}

func New() *Net {
	return &Net{}
}

func (n *Net) Set(layer ...layer.Layer) {
	n.layers = layer
}

func (n *Net) Forward(input *tensor.Tensor, isTraining bool) *tensor.Tensor {
	for i := 0; i < len(n.layers); i++ {
		input = n.layers[i].Forward(input, isTraining)
	}
	return input
}

func (n *Net) Params() []*params.Params {
	ret := make([]*params.Params, len(n.layers))
	for i := 0; i < len(n.layers); i++ {
		ret[i] = n.layers[i].Params()
	}
	return ret
}
