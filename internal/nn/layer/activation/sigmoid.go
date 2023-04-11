package activation

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Sigmoid struct {
	*base
}

func NewSigmoid() *Sigmoid {
	var sm Sigmoid
	sm.base = new(sm.activation, sm.derivative)
	return &sm
}

func (layer *Sigmoid) Name() string {
	return "sigmoid"
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (layer *Sigmoid) activation(input *mat.Dense) *mat.Dense {
	var ret mat.Dense
	ret.Apply(func(_, _ int, v float64) float64 {
		return sigmoid(v)
	}, &layer.input)
	return &ret
}

func (layer *Sigmoid) derivative(grad *mat.Dense) *mat.Dense {
	var ret mat.Dense
	ret.Apply(func(i, j int, v float64) float64 {
		return sigmoid(v) * (1 - sigmoid(v))
	}, &layer.input)
	return &ret
}
