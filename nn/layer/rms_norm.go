package layer

import (
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

func LoadRMSNorm(name string, params map[string]*tensor.Tensor, args map[string]float32) Layer {
	var layer RMSNorm
	layer.new("rms_norm", name)
	layer.paramType = params["a"].ScalarType()
	layer.eps = layer.initN(1e-9)
	layer.a = params["a"]
	return &layer
}

func (l *RMSNorm) norm(x *tensor.Tensor) *tensor.Tensor {
	return x.Mul(x.Pow(2).Mean(-1, true).Add(l.eps.ToDevice(x.DeviceType())).RSqrt())
}

func (layer *RMSNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	return layer.a.Mul(layer.norm(x))
}

func (layer *RMSNorm) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"a": layer.a,
	}
}

func (layer *RMSNorm) Freeze() {
	layer.a.SetRequiresGrad(false)
}

func (layer *RMSNorm) Unfreeze() {
	layer.a.SetRequiresGrad(true)
}
