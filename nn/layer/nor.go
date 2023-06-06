package layer

import (
	"github.com/lwch/runtime"
	"github.com/lwch/tnn/internal/pb"
	"gorgonia.org/gorgonia"
)

type Nor struct {
	*base
}

func NewNor(output int) *Nor {
	var layer Nor
	layer.base = new("nor")
	return &layer
}

func LoadNor(g *gorgonia.ExprGraph, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Nor
	layer.base = new("nor")
	layer.name = name
	return &layer
}

func (layer *Nor) Forward(x *gorgonia.Node) *gorgonia.Node {
	y, _, _, _, err := gorgonia.BatchNorm(x, nil, nil, 0.99, 1e-9)
	runtime.Assert(err)
	return y
}
