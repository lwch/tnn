package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type Dense struct {
	*base
	output int
	// params
	w *tensor.Tensor
}

func NewDense(output int, device consts.DeviceType) *Dense {
	var layer Dense
	layer.base = new("dense", device)
	layer.output = output
	return &layer
}

func LoadDense(device consts.DeviceType, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Dense
	layer.base = new("dense", device)
	layer.name = name
	layer.output = int(args["output"])
	layer.w = layer.loadParam(params["w"])
	return &layer
}

func (layer *Dense) Forward(x *tensor.Tensor) *tensor.Tensor {
	inputShape := x.Shapes()
	if layer.w == nil {
		layer.w = layer.initW(inputShape[len(inputShape)-1], int64(layer.output))
	}
	return x.MatMul(layer.w)
}

func (layer *Dense) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": layer.w,
	}
}

func (layer *Dense) Args() map[string]float32 {
	return map[string]float32{
		"output": float32(layer.output),
	}
}

func (layer *Dense) Freeze() {
	if layer.w != nil {
		layer.w.SetRequiresGrad(false)
	}
}

func (layer *Dense) Unfreeze() {
	if layer.w != nil {
		layer.w.SetRequiresGrad(true)
	}
}
