package activation

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Softplus struct {
	*base
}

func NewSoftplus() Activation {
	var layer Softplus
	layer.base = new("softplus", layer.activation, layer.derivative)
	return &layer
}

func softplus(x float64) float64 {
	return math.Log(1+math.Exp(-math.Abs(x))) + math.Max(x, 0)
}

func (layer *Softplus) activation(input *mat.Dense) *mat.Dense {
	var ret mat.Dense
	ret.Apply(func(_, _ int, v float64) float64 {
		return softplus(v)
	}, &layer.input)
	return &ret
}

func (layer *Softplus) derivative(grad *mat.Dense) *mat.Dense {
	var ret mat.Dense
	ret.Apply(func(i, j int, v float64) float64 {
		return sigmoid(v)
	}, &layer.input)
	return &ret
}
