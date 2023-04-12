package layer

import (
	"tnn/internal/initializer"
	"tnn/internal/nn/pb"

	"gonum.org/v1/gonum/mat"
)

type Dropout struct {
	*base
}

func NewDropout(keepProb float64) *Dropout {
	var layer Dropout
	layer.base = new(map[string]shape{
		"m": {noneShape, noneShape},
	}, initializer.NewBinomial(1, keepProb), layer.forward, layer.backward)
	layer.params["kp"] = mat.NewDense(1, 1, []float64{keepProb})
	return &layer
}

func LoadDropout(params map[string]*pb.Dense) Layer {
	var layer Dropout
	kp := params["kp"].GetData()[0]
	layer.base = new(nil, initializer.NewBinomial(1, kp),
		layer.forward, layer.backward)
	layer.base.loadParams(params)
	return &layer
}

func (layer *Dropout) Name() string {
	return "dropout"
}

func (layer *Dropout) forward(input *mat.Dense) *mat.Dense {
	if !layer.hasInit {
		rows, cols := input.Dims()
		layer.shapes["m"] = shape{rows, cols}
		layer.initParams()
	}
	kp := layer.params["kp"].At(0, 0)
	m := layer.context["m"]
	m.Apply(func(i, j int, v float64) float64 {
		return layer.init.Rand() / kp
	}, m)
	var ret mat.Dense
	ret.Apply(func(i, j int, v float64) float64 {
		return v * m.At(i, j)
	}, input)
	return &ret
}

func (layer *Dropout) backward(grad *mat.Dense) *mat.Dense {
	dm := layer.context["m"]

	var ret mat.Dense
	ret.Apply(func(i, j int, v float64) float64 {
		return v * dm.At(i, j)
	}, grad)
	return &ret
}
