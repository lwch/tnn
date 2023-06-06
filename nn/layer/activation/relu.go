package activation

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/layer"
	"gorgonia.org/gorgonia"
)

type ReLU struct {
	*base
}

func NewReLU() *ReLU {
	var layer ReLU
	layer.base = new("relu")
	return &layer
}

func LoadRelu(g *gorgonia.ExprGraph, name string, _ map[string]*pb.Dense, _ map[string]float32) layer.Layer {
	var layer ReLU
	layer.base = new("relu")
	layer.name = name
	return &layer
}

func (layer *ReLU) Forward(x *gorgonia.Node) *gorgonia.Node {
	return gorgonia.Must(gorgonia.Rectify(x))
}
