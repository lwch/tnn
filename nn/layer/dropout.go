package layer

import (
	"tnn/initializer"
	"tnn/nn/pb"
	"tnn/nn/vector"

	"gonum.org/v1/gonum/mat"
)

type Dropout struct {
	*base
}

func NewDropout(keepProb float64) *Dropout {
	var layer Dropout
	layer.base = new("dropout", map[string]Shape{
		"m": {NoneShape, NoneShape},
	}, initializer.NewBinomial(1, keepProb), layer.forward, layer.backward)
	layer.params["kp"] = mat.NewDense(1, 1, []float64{keepProb})
	return &layer
}

func LoadDropout(name string, params map[string]*pb.Dense) Layer {
	var layer Dropout
	kp := params["kp"].GetData()[0]
	layer.base = new("dropout", nil, initializer.NewBinomial(1, kp),
		layer.forward, layer.backward)
	layer.name = name
	layer.base.loadParams(params)
	return &layer
}

func (layer *Dropout) Name() string {
	return "dropout"
}

func (layer *Dropout) forward(input mat.Matrix) mat.Matrix {
	if !layer.hasInit {
		rows, cols := input.Dims()
		layer.shapes["m"] = Shape{rows, cols}
		layer.initParams()
	}
	kp := layer.params["kp"].At(0, 0)
	m := layer.context["m"]
	m.(vector.Applyer).Apply(func(i, j int, v float64) float64 {
		return layer.init.Rand() / kp
	}, m)
	var ret mat.Dense
	ret.Apply(func(i, j int, v float64) float64 {
		return v * m.At(i, j)
	}, input)
	return &ret
}

func (layer *Dropout) backward(grad mat.Matrix) mat.Matrix {
	dm := layer.context["m"]

	var ret mat.Dense
	ret.Apply(func(i, j int, v float64) float64 {
		return v * dm.At(i, j)
	}, grad)
	return &ret
}
