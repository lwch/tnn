package activation

import (
	"tnn/internal/nn/layer"
	"tnn/internal/nn/params"
	"tnn/internal/nn/pb"

	"gonum.org/v1/gonum/mat"
)

type Activation interface {
	layer.Layer
}

func Load(name string) func(map[string]*pb.Dense) layer.Layer {
	switch name {
	case "sigmoid":
		return func(map[string]*pb.Dense) layer.Layer {
			return NewSigmoid()
		}
	default:
		return nil
	}
}

type activationFunc func(*mat.Dense) *mat.Dense
type derivativeFunc func(*mat.Dense) *mat.Dense

type base struct {
	input      mat.Dense
	activation activationFunc
	derivative derivativeFunc
}

func new(activation activationFunc, derivative derivativeFunc) *base {
	return &base{
		activation: activation,
		derivative: derivative,
	}
}

func (*base) Name() {
	panic("unimplemented")
}

func (layer *base) Forward(input *mat.Dense) *mat.Dense {
	layer.input.CloneFrom(input)
	return layer.activation(input)
}

func (layer *base) Backward(grad *mat.Dense) *mat.Dense {
	var ret mat.Dense
	ret.Apply(func(i, j int, v float64) float64 {
		return v * grad.At(i, j)
	}, layer.derivative(grad))
	return &ret
}

func (*base) Params() *params.Params {
	return nil
}

func (*base) Context() params.Params {
	return nil
}

func (*base) Active() {
	panic("unimplemented")
}

func (*base) Derivative() {
	panic("unimplemented")
}
