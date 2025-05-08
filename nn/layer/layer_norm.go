package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

type LayerNorm struct {
	base
	eps *tensor.Tensor
	// params
	a *tensor.Tensor
}

func NewLayerNorm(name string, dims int64, opts ...LayerCreateOption) *LayerNorm {
	var layer LayerNorm
	layer.new("layer_norm", name, opts...)
	layer.eps = layer.initN(1e-9)
	layer.a = layer.ones(dims)
	layer.a.SetRequiresGrad(true)
	return &layer
}

func LoadLayerNorm(name string, params []*tensor.Tensor, args map[string]float32) Layer {
	var layer LayerNorm
	layer.new("layer_norm", name)
	layer.eps = layer.initN(1e-9)
	layer.a = params[0]
	return &layer
}

func (layer *LayerNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	mean := x.Mean(-1, true)
	v := x.Var(-1, false, true)
	sub := x.Sub(mean)
	bias := v.Add(layer.eps).Sqrt()
	div := sub.Div(bias)
	return div.Mul(layer.a)
}

func (layer *LayerNorm) Params() []*tensor.Tensor {
	return []*tensor.Tensor{
		layer.a,
	}
}

func (layer *LayerNorm) Freeze() {
	layer.a.SetRequiresGrad(false)
}

func (layer *LayerNorm) Unfreeze() {
	layer.a.SetRequiresGrad(true)
}

func (layer *LayerNorm) ToScalarType(t consts.ScalarType) {
	layer.a = layer.a.ToScalarType(t)
}

func (layer *LayerNorm) Reset() {
	layer.a = layer.ones(layer.a.Shapes()...)
	layer.a.SetRequiresGrad(true)
}

func (layer *LayerNorm) Clone() Layer {
	return &LayerNorm{
		base: layer.base.clone(),
		eps:  layer.eps.Clone(),
		a:    layer.a.Clone(),
	}
}
