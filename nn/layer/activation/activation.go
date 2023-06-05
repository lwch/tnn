package activation

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/layer"
	"gorgonia.org/gorgonia"
)

func Load(class string) func(*gorgonia.ExprGraph, string, map[string]*pb.Dense, map[string]float32) layer.Layer {
	var fn func() layer.Layer
	switch class {
	// case "sigmoid":
	// 	fn = NewSigmoid
	// case "softplus":
	// 	fn = NewSoftplus
	// case "tanh":
	// 	fn = NewTanh
	case "relu":
		fn = NewReLU
	// case "gelu":
	// 	fn = NewGeLU
	default:
		return nil
	}
	return func(*gorgonia.ExprGraph, string, map[string]*pb.Dense, map[string]float32) layer.Layer {
		return fn()
	}
}

type base struct {
	class string
	name  string
}

func new(class, name string) *base {
	return &base{
		class: class,
		name:  name,
	}
}

func (layer *base) Class() string {
	return layer.class
}

func (layer *base) SetName(name string) {
	layer.name = name
}

func (layer *base) Name() string {
	if len(layer.name) == 0 {
		return layer.class
	}
	return layer.name
}

func (layer *base) Params() gorgonia.Nodes {
	return nil
}

func (*base) Args() map[string]float32 {
	return nil
}
