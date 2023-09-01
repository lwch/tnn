package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type LayerNorm struct {
	base
	eps *tensor.Tensor
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
	return &layer
}

func (layer *LayerNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	if layer.eps == nil {
		layer.eps = tensor.FromFloat32(nil, []float32{1e-9}, tensor.WithShapes(1), tensor.WithDevice(layer.device))
	}
	mean := x.Mean(-1, true)
	v := x.Var(-1, false, true)
	sub := x.Sub(mean)
	bias := v.Add(layer.eps)
	return sub.Div(bias)
}
