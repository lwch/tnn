package net

import (
	"tnn/internal/nn/layer"
	"tnn/internal/nn/params"

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

func (n *Net) Backward(grad *mat.Dense) []*params.Params {
	ret := make([]*params.Params, len(n.layers))
	for i := len(n.layers) - 1; i >= 0; i-- {
		grad = n.layers[i].Backward(grad)
		var p params.Params
		p.Copy(n.layers[i].Context())
		ret[i] = &p
	}
	return ret
}

func (n *Net) Params() []*params.Params {
	ret := make([]*params.Params, len(n.layers))
	for i := 0; i < len(n.layers); i++ {
		ret[i] = n.layers[i].Params()
	}
	return ret
}
