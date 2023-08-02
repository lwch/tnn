package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type LayerNorm struct {
	base
	eps *tensor.Tensor
	// params
	w *tensor.Tensor
	b *tensor.Tensor
}

func NewLayerNorm(opts ...LayerCreateOption) *LayerNorm {
	var layer LayerNorm
	layer.new("layer_norm", opts...)
	return &layer
}

func LoadLayerNorm(device consts.DeviceType, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer LayerNorm
	layer.new("layer_norm", WithDevice(device))
	layer.name = name
	layer.w = layer.loadParam(params["w"])
	layer.b = layer.loadParam(params["b"])
	return &layer
}

func (layer *LayerNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	if layer.eps == nil {
		layer.eps = tensor.FromFloat32(nil, []float32{1e-9}, tensor.WithShapes(1), tensor.WithDevice(layer.device))
	}
	if layer.w == nil {
		layer.w = layer.Ones(lastDim(x))
		layer.w.SetRequiresGrad(true)
	}
	if layer.b == nil {
		layer.b = layer.initB(lastDim(x))
	}
	mean := x.Mean(-1, true)
	v := x.Var(-1, false, true)
	return layer.w.Mul(x.Sub(mean)).Div(v.Add(layer.eps).Sqrt()).Add(layer.b)
}

func (layer *LayerNorm) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": layer.w,
		"b": layer.b,
	}
}

func (layer *LayerNorm) Freeze() {
	if layer.w != nil {
		layer.w.SetRequiresGrad(false)
	}
	if layer.b != nil {
		layer.b.SetRequiresGrad(false)
	}
}

func (layer *LayerNorm) Unfreeze() {
	if layer.w != nil {
		layer.w.SetRequiresGrad(true)
	}
	if layer.b != nil {
		layer.b.SetRequiresGrad(true)
	}
}
