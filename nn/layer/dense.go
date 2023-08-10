package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type Dense struct {
	base
	output int
	bias   bool
	// params
	w *tensor.Tensor
	b *tensor.Tensor
}

func NewDense(output int, bias bool, opts ...LayerCreateOption) *Dense {
	var layer Dense
	layer.new("dense", opts...)
	layer.output = output
	layer.bias = bias
	return &layer
}

func LoadDense(device consts.DeviceType, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Dense
	layer.new("dense", WithDevice(device))
	layer.name = name
	layer.output = int(args["output"])
	layer.bias = args["bias"] > 0
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
	x = x.MatMul(layer.w)
	if layer.bias {
		x = x.Add(layer.b)
	}
	return x
}

func (layer *Dense) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": layer.w,
		"b": layer.b,
	}
}

func (layer *Dense) Args() map[string]float32 {
	var bias float32
	if layer.bias {
		bias = 1
	}
	return map[string]float32{
		"output": float32(layer.output),
		"bias":   bias,
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
