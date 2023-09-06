package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type Linear struct {
	base
	output int
	// params
	w *tensor.Tensor
	b *tensor.Tensor
}

func NewLinear(output int, opts ...LayerCreateOption) *Linear {
	var layer Linear
	layer.new("linear", opts...)
	layer.output = output
	return &layer
}

func LoadLinear(device consts.DeviceType, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Linear
	layer.new("linear", WithDevice(device))
	layer.name = name
	layer.output = int(args["output"])
	layer.w = layer.loadParam(params["w"])
	layer.b = layer.loadParam(params["b"])
	return &layer
}

func (layer *Linear) Forward(x *tensor.Tensor) *tensor.Tensor {
	inputShape := x.Shapes()
	if layer.w == nil {
		layer.w = layer.initW(inputShape[len(inputShape)-1], int64(layer.output))
	}
	if layer.b == nil {
		layer.b = layer.initB(int64(layer.output))
	}
	return x.MatMul(layer.w).Add(layer.b)
}

func (layer *Linear) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": layer.w,
		"b": layer.b,
	}
}

func (layer *Linear) Args() map[string]float32 {
	return map[string]float32{
		"output": float32(layer.output),
	}
}

func (layer *Linear) Freeze() {
	if layer.w != nil {
		layer.w.SetRequiresGrad(false)
	}
	if layer.b != nil {
		layer.b.SetRequiresGrad(false)
	}
}

func (layer *Linear) Unfreeze() {
	if layer.w != nil {
		layer.w.SetRequiresGrad(true)
	}
	if layer.b != nil {
		layer.b.SetRequiresGrad(true)
	}
}
