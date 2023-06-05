package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Flatten struct {
	*base
}

func NewFlatten(g *gorgonia.ExprGraph) Layer {
	var layer Flatten
	layer.base = new("flatten")
	return &layer
}

func LoadFlatten(g *gorgonia.ExprGraph, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Flatten
	layer.base = new("flatten")
	layer.name = name
	return &layer
}

func (layer *Flatten) Forward(x *gorgonia.Node) *gorgonia.Node {
	shape := x.Shape()
	cols := 1
	for _, v := range shape[1:] {
		cols *= v
	}
	return gorgonia.Must(gorgonia.Reshape(x, tensor.Shape{shape[0], cols}))
}
