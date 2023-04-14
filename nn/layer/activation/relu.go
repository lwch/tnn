package activation

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type ReLU struct {
	*base
}

func NewReLU() Activation {
	var layer ReLU
	layer.base = new("relu", layer.activation, layer.derivative)
	return &layer
}

func relu(x float64) float64 {
	return math.Max(x, 0)
}

func (layer *ReLU) activation(input mat.Matrix) *mat.Dense {
	var ret mat.Dense
	ret.Apply(func(_, _ int, v float64) float64 {
		return relu(v)
	}, &layer.input)
	return &ret
}

func (layer *ReLU) derivative() *mat.Dense {
	var ret mat.Dense
	ret.Apply(func(i, j int, v float64) float64 {
		if v > 0 {
			return 1
		}
		return 0
	}, &layer.input)
	return &ret
}
