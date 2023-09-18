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

func NewLinear(input, output int, opts ...LayerCreateOption) *Linear {
	var layer Linear
	layer.new("linear", opts...)
	layer.output = output
	layer.w = layer.initW(int64(input), int64(layer.output))
	return &layer
}

func LoadLinear(device consts.DeviceType, name string, params map[string]*tensor.Tensor, args map[string]float32) Layer {
	var layer Linear
	layer.new("linear", WithDevice(device))
	layer.name = name
	layer.output = int(args["output"])
	layer.w = params["w"]
	return &layer
}

func (layer *Linear) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.MatMul(layer.w)
}

func (layer *Linear) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": layer.w,
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
