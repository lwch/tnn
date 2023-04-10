package net

import (
	"tnn/internal/nn/layer"

	"gonum.org/v1/gonum/mat"
)

type Net struct {
	layers []layer.Layer
}

func New() *Net {
	return &Net{}
}

func (n *Net) Add(layer ...layer.Layer) {
	n.layers = append(n.layers, layer...)
}

func (n *Net) Forward(input *mat.Dense) *mat.Dense {
	for i := 0; i < len(n.layers); i++ {
		input = n.layers[i].Forward(input)
	}
	return input
}

func (n *Net) Backward(grad *mat.Dense) []*layer.Params {
	ret := make([]*layer.Params, len(n.layers))
	for i := len(n.layers) - 1; i >= 0; i-- {
		grad = n.layers[i].Backward(grad)
		ret[i] = new(layer.Params)
		ret[i].Copy(n.layers[i].Context())
	}
	return ret
}

func (n *Net) Params() []*layer.Params {
	ret := make([]*layer.Params, len(n.layers))
	for i := 0; i < len(n.layers); i++ {
		ret[i] = n.layers[i].Params()
	}
	return ret
}
