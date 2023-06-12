package layer

import (
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type Dense struct {
	*base
	output int
	// params
	w *tensor.Tensor
	b *tensor.Tensor
}

func NewDense(output int) *Dense {
	var layer Dense
	layer.base = new("dense")
	layer.output = output
	return &layer
}

func LoadDense(name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Dense
	layer.base = new("dense")
	layer.name = name
	layer.output = int(args["output"])
	layer.w = loadParam(params["w"])
	layer.b = loadParam(params["b"])
	return &layer
}

func (layer *Dense) Forward(x *tensor.Tensor) *tensor.Tensor {
	inputShape := x.Shapes()
	if layer.w == nil {
		layer.w = initW(inputShape[1], int64(layer.output))
	}
	if layer.b == nil {
		layer.b = initB(int64(layer.output))
	}
	return x.MatMul(layer.w).Add(layer.b)
}

func (layer *Dense) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": layer.w,
		"b": layer.b,
	}
}

func (layer *Dense) Args() map[string]float32 {
	return map[string]float32{
		"output": float32(layer.output),
	}
}
