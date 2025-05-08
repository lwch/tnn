package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

type Linear struct {
	base
	output int
	// params
	w *tensor.Tensor
}

func NewLinear(name string, input, output int, opts ...LayerCreateOption) *Linear {
	var layer Linear
	layer.new("linear", name, opts...)
	layer.output = output
	layer.w = layer.initW(int64(layer.output), int64(input))
	return &layer
}

func LoadLinear(name string, params []*tensor.Tensor, args map[string]float32) Layer {
	var layer Linear
	layer.new("linear", name)
	layer.output = int(args["output"])
	layer.w = params[0]
	return &layer
}

func (layer *Linear) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.MatMul(layer.w.Transpose(0, 1))
}

func (layer *Linear) Params() []*tensor.Tensor {
	return []*tensor.Tensor{
		layer.w,
	}
}

func (layer *Linear) Args() map[string]float32 {
	return map[string]float32{
		"output": float32(layer.output),
	}
}

func (layer *Linear) Freeze() {
	layer.w.SetRequiresGrad(false)
}

func (layer *Linear) Unfreeze() {
	layer.w.SetRequiresGrad(true)
}

func (layer *Linear) ToScalarType(t consts.ScalarType) {
	layer.w = layer.w.ToScalarType(t)
}

func (layer *Linear) Reset() {
	layer.w = layer.initW(layer.w.Shapes()...)
}

func (layer *Linear) Clone() Layer {
	return &Linear{
		base:   layer.base.clone(),
		output: layer.output,
		w:      layer.w.Clone(),
	}
}
