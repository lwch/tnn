package activation

import (
	"github.com/lwch/gonum/mat32"
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/params"
)

type Activation interface {
	layer.Layer
}

func Load(class string) func(string, map[string]*pb.Dense, map[string]*pb.Dense) layer.Layer {
	var fn func() Activation
	switch class {
	case "sigmoid":
		fn = NewSigmoid
	// case "softplus":
	// 	fn = NewSoftplus
	// case "tanh":
	// 	fn = NewTanh
	case "relu":
		fn = NewReLU
	case "gelu":
		fn = NewGeLU
	default:
		return nil
	}
	return func(name string, _ map[string]*pb.Dense, _ map[string]*pb.Dense) layer.Layer {
		return fn()
	}
}

type base struct {
	class string
	name  string
}

func new(class string) *base {
	return &base{
		class: class,
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

func (layer *base) Params() *params.Params {
	return nil
}

func (*base) Args() map[string]*mat32.VecDense {
	return nil
}
