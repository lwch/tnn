package layer

import (
	"tnn/internal/initializer"
	"tnn/internal/nn/pb"

	"gonum.org/v1/gonum/mat"
)

type Dense struct {
	*base
}

func NewDense(output int, init initializer.Initializer) *Dense {
	var layer Dense
	layer.base = new(map[string]shape{
		"w": {noneShape, output}, // rows reshape from input
		"b": {noneShape, output}, // rows reshape from input
	}, init, layer.forward, layer.backward)
	return &layer
}

func LoadDense(params map[string]*pb.Dense) Layer {
	var layer Dense
	layer.base = new(nil, nil, layer.forward, layer.backward)
	layer.base.loadParams(params)
	return &layer
}

func (layer *Dense) Name() string {
	return "dense"
}

func (layer *Dense) forward(input *mat.Dense) *mat.Dense {
	if !layer.hasInit {
		shapeW := layer.shapes["w"]
		shapeB := layer.shapes["b"]
		_, shapeW.m = input.Dims()
		shapeB.m, _ = input.Dims()
		layer.shapes["w"] = shapeW
		layer.shapes["b"] = shapeB
		layer.initParams()
	}
	var ret mat.Dense
	ret.Mul(input, layer.params["w"])
	ret.Add(&ret, layer.params["b"])
	return &ret
}

func (layer *Dense) backward(grad *mat.Dense) *mat.Dense {
	dw := layer.context["w"]
	db := layer.context["b"]

	dw.Mul(layer.input.T(), grad)
	db.Copy(grad)

	var ret mat.Dense
	w := layer.params["w"]
	ret.Mul(grad, w.T())
	return &ret
}
