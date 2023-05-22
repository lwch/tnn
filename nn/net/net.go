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

func (n *Net) Forward(input *tensor.Tensor, watchList *params.List, isTraining bool) *tensor.Tensor {
	for i := 0; i < len(n.layers); i++ {
		input = n.layers[i].Forward(input, watchList, isTraining)
	}
	return input
}
