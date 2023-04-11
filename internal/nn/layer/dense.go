package layer

import (
	"tnn/internal/initializer"

	"gonum.org/v1/gonum/mat"
)

type Dense struct {
	*base
}

func NewDense(output int, init initializer.Initializer) *Dense {
	var d Dense
	d.base = new(map[string]shape{
		"w": {noneShape, output}, // rows reshape from input
		"b": {1, output},
	}, init, d.forward, d.backward)
	return &d
}

func (layer *Dense) Name() string {
	return "dense"
}

func (layer *Dense) forward(input *mat.Dense) *mat.Dense {
	if !layer.hasInit {
		shape := layer.shapes["w"]
		_, shape.m = input.Dims()
		layer.shapes["w"] = shape
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
