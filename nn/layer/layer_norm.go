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
	b *tensor.Tensor
}

func NewLayerNorm(dims int64, opts ...LayerCreateOption) *LayerNorm {
	var layer LayerNorm
	layer.new("layer_norm", opts...)
	data := make([]float32, dims)
	for i := range data {
		data[i] = 1
	}
	layer.eps = tensor.FromFloat32(nil, []float32{1e-9},
		tensor.WithShapes(1),
		tensor.WithDevice(layer.device))
	layer.a = tensor.FromFloat32(nil, data,
		tensor.WithShapes(dims),
		tensor.WithDevice(layer.device))
	layer.a.SetRequiresGrad(true)
	layer.b = layer.initB(dims)
	return &layer
}

func LoadLayerNorm(device consts.DeviceType, name string, params map[string]*tensor.Tensor, args map[string]float32) Layer {
	var layer LayerNorm
	layer.new("layer_norm", WithDevice(device))
	layer.name = name
	layer.eps = tensor.FromFloat32(nil, []float32{1e-9},
		tensor.WithShapes(1),
		tensor.WithDevice(layer.device))
	layer.a = params["a"]
	layer.b = params["b"]
	return &layer
}

func (layer *LayerNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	mean := x.Mean(-1, true)
	v := x.Var(-1, false, true)
	sub := x.Sub(mean)
	bias := v.Add(layer.eps).Sqrt()
	div := sub.Div(bias)
	return div.Mul(layer.a).Add(layer.b)
}

func (layer *LayerNorm) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"a": layer.a,
		"b": layer.b,
	}
}

func (layer *LayerNorm) Freeze() {
	layer.a.SetRequiresGrad(false)
	layer.b.SetRequiresGrad(false)
}

func (layer *LayerNorm) Unfreeze() {
	layer.a.SetRequiresGrad(true)
	layer.b.SetRequiresGrad(true)
}
