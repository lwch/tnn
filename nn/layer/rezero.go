package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type ReZero struct {
	base
	// params
	scale *tensor.Tensor
}

func NewReZero(opts ...LayerCreateOption) *ReZero {
	var layer ReZero
	layer.new("rezero", opts...)
	return &layer
}

func LoadReZero(device consts.DeviceType, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer ReZero
	layer.new("rezero", WithDevice(device))
	layer.name = name
	layer.scale = layer.loadParam(params["scale"])
	return &layer
}

func (layer *ReZero) Forward(x *tensor.Tensor) *tensor.Tensor {
	if layer.scale == nil {
		layer.scale = tensor.FromFloat32(x.Storage(), []float32{0},
			tensor.WithShapes(1),
			tensor.WithDevice(layer.device))
		layer.scale.SetRequiresGrad(true)
	}
	return x.Mul(layer.scale)
}

func (layer *ReZero) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"scale": layer.scale,
	}
}

func (layer *ReZero) Args() map[string]float32 {
	return map[string]float32{}
}

func (layer *ReZero) Freeze() {
	if layer.scale != nil {
		layer.scale.SetRequiresGrad(false)
	}
}

func (layer *ReZero) Unfreeze() {
	if layer.scale != nil {
		layer.scale.SetRequiresGrad(true)
	}
}
