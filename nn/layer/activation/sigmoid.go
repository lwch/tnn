package activation

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Sigmoid struct {
	*base
}

func NewSigmoid() Activation {
	var layer Sigmoid
	layer.base = new("sigmoid", layer.activation, layer.derivative)
	return &layer
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (layer *Sigmoid) activation(input mat.Matrix) *mat.Dense {
	var ret mat.Dense
	ret.Apply(func(_, _ int, v float64) float64 {
		return sigmoid(v)
	}, &layer.input)
	return &ret
}

func (layer *Sigmoid) derivative() *mat.Dense {
	var ret mat.Dense
	ret.Apply(func(i, j int, v float64) float64 {
		return sigmoid(v) * (1 - sigmoid(v))
	}, &layer.input)
	return &ret
}
