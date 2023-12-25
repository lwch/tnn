package layer

import (
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
	data := make([]float32, dims)
	for i := range data {
		data[i] = 1
	}
	layer.eps = tensor.FromFloat32(name+".eps", []float32{1e-9},
		tensor.WithShapes(1),
		tensor.WithDevice(layer.device))
	layer.a = tensor.FromFloat32(name+".a", data,
		tensor.WithShapes(dims),
		tensor.WithDevice(layer.device))
	layer.a.SetRequiresGrad(true)
	return &layer
}

func LoadLayerNorm(name string, params map[string]*tensor.Tensor, args map[string]float32) Layer {
	var layer LayerNorm
	layer.new("layer_norm", name)
	layer.eps = tensor.FromFloat32(name+".eps", []float32{1e-9},
		tensor.WithShapes(1),
		tensor.WithDevice(layer.device))
	layer.a = params["a"]
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

func (layer *LayerNorm) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"a": layer.a,
	}
}

func (layer *LayerNorm) Freeze() {
	layer.a.SetRequiresGrad(false)
}

func (layer *LayerNorm) Unfreeze() {
	layer.a.SetRequiresGrad(true)
}
