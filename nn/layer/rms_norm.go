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
	data := make([]float32, dims)
	for i := range data {
		data[i] = 1
	}
	layer.eps = tensor.FromFloat32([]float32{1e-9},
		tensor.WithShapes(1),
		tensor.WithDevice(layer.device))
	layer.a = tensor.FromFloat32(data,
		tensor.WithShapes(dims),
		tensor.WithDevice(layer.device))
	layer.a.SetRequiresGrad(true)
	return &layer
}

func LoadRMSNorm(name string, params map[string]*tensor.Tensor, args map[string]float32) Layer {
	var layer RMSNorm
	layer.new("rms_norm", name)
	layer.eps = tensor.FromFloat32([]float32{1e-9},
		tensor.WithShapes(1),
		tensor.WithDevice(layer.device))
	layer.a = params["a"]
	return &layer
}

func (l *RMSNorm) norm(x *tensor.Tensor) *tensor.Tensor {
	return x.Mul(x.Pow(2).Mean(-1, true).Add(l.eps).RSqrt())
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
