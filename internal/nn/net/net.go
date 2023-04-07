package net

import (
	"tnn/internal/nn/layer"
	"tnn/internal/nn/optimizer"

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

func (n *Net) Backward(grad *mat.Dense) {
	for i := len(n.layers) - 1; i >= 0; i-- {
		grad = n.layers[i].Backward(grad)
	}
}

func (n *Net) Update(optimizer optimizer.Optimizer) {
	for i := len(n.layers) - 1; i >= 0; i-- {
		n.layers[i].Update(optimizer)
	}
}
