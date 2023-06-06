package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"gorgonia.org/gorgonia"
)

type Dropout struct {
	*base
	keep float64
}

func NewDropout(g *gorgonia.ExprGraph, keep float64) *Dropout {
	var layer Dropout
	layer.base = new("dropout")
	layer.keep = keep
	return &layer
}

func LoadDropout(g *gorgonia.ExprGraph, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Dropout
	layer.base = new("dropout")
	layer.name = name
	layer.keep = float64(args["keep"])
	return &layer
}

func (layer *Dropout) Forward(x *gorgonia.Node, train bool) *gorgonia.Node {
	if !train {
		return x
	}
	return gorgonia.Must(gorgonia.Dropout(x, layer.keep))
}

func (layer *Dropout) Args() map[string]float32 {
	return map[string]float32{
		"keep": float32(layer.keep),
	}
}
