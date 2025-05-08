package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

type RMSNorm struct {
	base
	eps *tensor.Tensor
	// params
	a *tensor.Tensor
}

func NewRMSNorm(name string, dims int64, opts ...LayerCreateOption) *RMSNorm {
	var layer RMSNorm
	layer.new("rms_norm", name, opts...)
	layer.eps = layer.initN(1e-9)
	layer.a = layer.ones(dims)
	layer.a.SetRequiresGrad(true)
	return &layer
}

func LoadRMSNorm(name string, params []*tensor.Tensor, args map[string]float32) Layer {
	var layer RMSNorm
	layer.new("rms_norm", name)
	layer.paramType = params[0].ScalarType()
	layer.eps = layer.initN(1e-9)
	layer.a = params[0]
	return &layer
}

func (l *RMSNorm) norm(x *tensor.Tensor) *tensor.Tensor {
	return x.Mul(x.Pow(2).Mean(-1, true).Add(l.eps.ToDevice(x.DeviceType())).RSqrt())
}

func (layer *RMSNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	return layer.a.Mul(layer.norm(x))
}

func (layer *RMSNorm) Params() []*tensor.Tensor {
	return []*tensor.Tensor{
		layer.a,
	}
}

func (layer *RMSNorm) Freeze() {
	layer.a.SetRequiresGrad(false)
}

func (layer *RMSNorm) Unfreeze() {
	layer.a.SetRequiresGrad(true)
}

func (layer *RMSNorm) ToScalarType(t consts.ScalarType) {
	layer.a = layer.a.ToScalarType(t)
}

func (layer *RMSNorm) Reset() {
	layer.a = layer.ones(layer.a.Shapes()...)
	layer.a.SetRequiresGrad(true)
}

func (layer *RMSNorm) Clone() Layer {
	return &RMSNorm{
		base: layer.base.clone(),
		eps:  layer.eps.Clone(),
		a:    layer.a.Clone(),
	}
}
