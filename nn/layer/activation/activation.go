package activation

import (
	"fmt"
	"tnn/nn/layer"
	"tnn/nn/params"
	"tnn/nn/pb"

	"gonum.org/v1/gonum/mat"
)

type Activation interface {
	layer.Layer
}

func Load(name string) func(map[string]*pb.Dense) layer.Layer {
	var fn Activation
	switch name {
	case "sigmoid":
		fn = NewSigmoid()
	case "softplus":
		fn = NewSoftplus()
	default:
		return nil
	}
	return func(map[string]*pb.Dense) layer.Layer {
		return fn
	}
}

type activationFunc func(*mat.Dense) *mat.Dense
type derivativeFunc func(*mat.Dense) *mat.Dense

type base struct {
	name       string
	input      mat.Dense
	activation activationFunc
	derivative derivativeFunc
}

func new(name string, activation activationFunc, derivative derivativeFunc) *base {
	return &base{
		name:       name,
		activation: activation,
		derivative: derivative,
	}
}

func (layer *base) Name() string {
	return layer.name
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

func (layer *base) Print() {
	fmt.Println("  - Name:", layer.Name())
}
