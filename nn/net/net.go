package net

import (
	"github.com/lwch/tnn/nn/layer"
	"gorgonia.org/gorgonia"
)

type Net struct {
	layers []layer.Layer
}

func New(layers ...layer.Layer) *Net {
	return &Net{
		layers: layers,
	}
}

func (n *Net) Add(layers ...layer.Layer) {
	n.layers = append(n.layers, layers...)
}

func (n *Net) Forward(x *gorgonia.Node) *gorgonia.Node {
	for _, l := range n.layers {
		x = l.Forward(x)
	}
	return x
}

func (n *Net) Params() gorgonia.Nodes {
	var ret gorgonia.Nodes
	for _, l := range n.layers {
		ret = append(ret, l.Params()...)
	}
	return ret
}

func (n *Net) ParamCount() int {
	var ret int
	for _, l := range n.layers {
		for _, p := range l.Params() {
			ret += p.Shape().TotalSize()
		}
	}
	return ret
}
