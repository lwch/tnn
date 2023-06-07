package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

type Dense struct {
	*base
	output int
	// params
	w *ts.Tensor
	b *ts.Tensor
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

func (layer *Dense) Forward(vs *nn.Path, x *ts.Tensor) *ts.Tensor {
	inputShape := x.MustSize()
	if layer.w == nil {
		layer.w = initW(vs, "w", inputShape[1], int64(layer.output))
	}
	if layer.b == nil {
		layer.b = initB(vs, "b", inputShape[0], int64(layer.output))
	}
	return x.MustMm(layer.w, false).MustAdd(layer.b, true)
}

func (layer *Dense) Params() map[string]*ts.Tensor {
	return map[string]*ts.Tensor{
		"w": layer.w,
		"b": layer.b,
	}
}

func (layer *Dense) Args() map[string]float32 {
	return map[string]float32{
		"output": float32(layer.output),
	}
}
