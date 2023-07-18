package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type Dense struct {
	base
	output int
	// params
	w *tensor.Tensor
	b *tensor.Tensor
}

func NewDense(output int, opts ...LayerCreateOption) *Dense {
	var layer Dense
	layer.new("dense", opts...)
	layer.output = output
	return &layer
}

func LoadDense(device consts.DeviceType, name string, params map[string]*pb.Dense, args map[string]float64) Layer {
	var layer Dense
	layer.new("dense", WithDevice(device))
	layer.name = name
	layer.output = int(args["output"])
	layer.w = layer.loadParam(params["w"])
	layer.b = layer.loadParam(params["b"])
	return &layer
}

func (layer *Dense) Forward(x *tensor.Tensor) *tensor.Tensor {
	inputShape := x.Shapes()
	if layer.w == nil {
		layer.w = layer.initW(inputShape[len(inputShape)-1], int64(layer.output))
	}
	if layer.b == nil {
		layer.b = layer.initB(int64(layer.output))
	}
	return x.MatMul(layer.w).Add(layer.b)
}

func (layer *Dense) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": layer.w,
		"b": layer.b,
	}
}

func (layer *Dense) Args() map[string]float64 {
	return map[string]float64{
		"output": float64(layer.output),
	}
}

func (layer *Dense) Freeze() {
	if layer.w != nil {
		layer.w.SetRequiresGrad(false)
	}
	if layer.b != nil {
		layer.b.SetRequiresGrad(false)
	}
}

func (layer *Dense) Unfreeze() {
	if layer.w != nil {
		layer.w.SetRequiresGrad(true)
	}
	if layer.b != nil {
		layer.b.SetRequiresGrad(true)
	}
}
