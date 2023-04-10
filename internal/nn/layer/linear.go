package layer

import (
	"tnn/internal/initializer"

	"gonum.org/v1/gonum/mat"
)

type Linear struct {
	*base
}

func NewLinear(output int, init initializer.Initializer) *Linear {
	return &Linear{
		base: new(map[string]shape{
			"w": {noneShape, output}, // rows reshape from input
			"b": {1, output},
		}, init),
	}
}

func (layer *Linear) Name() string {
	return "linear"
}

func (layer *Linear) Forward(input *mat.Dense) *mat.Dense {
	layer.input.CloneFrom(input)
	if !layer.hasInit {
		shape := layer.shapes["w"]
		_, shape.m = input.Dims()
		layer.shapes["w"] = shape
		layer.initParams(layer.init)
	}
	var ret mat.Dense
	ret.Mul(input, layer.params["w"])
	ret.Add(&ret, layer.params["b"])
	return &ret
}

func (layer *Linear) Backward(grad *mat.Dense) *mat.Dense {
	dw := layer.context["w"]
	db := layer.context["b"]

	dw.Mul(layer.input.T(), grad)
	db.Add(db, grad)

	var ret mat.Dense
	w := layer.params["w"]
	ret.Mul(grad, w.T())
	return &ret
}
