package layer

import (
	"github.com/lwch/gotorch/tensor"
)

type ReZero struct {
	base
	// params
	scale *tensor.Tensor
}

func NewReZero(opts ...LayerCreateOption) *ReZero {
	var layer ReZero
	layer.new("rezero", opts...)
	layer.scale = tensor.FromFloat32(nil, []float32{0},
		tensor.WithShapes(1),
		tensor.WithDevice(layer.device))
	layer.scale.SetRequiresGrad(true)
	return &layer
}

func LoadReZero(name string, params map[string]*tensor.Tensor, args map[string]float32) Layer {
	var layer ReZero
	layer.new("rezero")
	layer.name = name
	layer.scale = params["scale"]
	return &layer
}

func (layer *ReZero) Forward(x *tensor.Tensor) *tensor.Tensor {
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
	layer.scale.SetRequiresGrad(false)
}

func (layer *ReZero) Unfreeze() {
	layer.scale.SetRequiresGrad(true)
}
