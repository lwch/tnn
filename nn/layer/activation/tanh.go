package activation

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Tanh struct {
	*base
}

func NewTanh() Activation {
	var layer Tanh
	layer.base = new("tanh", layer.activation, layer.derivative)
	return &layer
}

func tanh(x float64) float64 {
	return math.Tanh(x)
}

func (layer *Tanh) activation(input mat.Matrix) *mat.Dense {
	var ret mat.Dense
	ret.Apply(func(_, _ int, v float64) float64 {
		return tanh(v)
	}, &layer.input)
	return &ret
}

func (layer *Tanh) derivative() *mat.Dense {
	var ret mat.Dense
	ret.Apply(func(i, j int, v float64) float64 {
		return 1 - math.Pow(tanh(v), 2)
	}, &layer.input)
	return &ret
}
