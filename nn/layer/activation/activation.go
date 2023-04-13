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

func Load(class string) func(string, map[string]*pb.Dense) layer.Layer {
	var fn func() Activation
	switch class {
	case "sigmoid":
		fn = NewSigmoid
	case "softplus":
		fn = NewSoftplus
	case "tanh":
		fn = NewTanh
	default:
		return nil
	}
	return func(name string, _ map[string]*pb.Dense) layer.Layer {
		return fn()
	}
}

type activationFunc func(*mat.Dense) *mat.Dense
type derivativeFunc func() *mat.Dense

type base struct {
	class      string
	name       string
	input      mat.Dense
	activation activationFunc
	derivative derivativeFunc
}

func new(class string, activation activationFunc, derivative derivativeFunc) *base {
	return &base{
		class:      class,
		activation: activation,
		derivative: derivative,
	}
}

func (layer *base) Class() string {
	return layer.class
}

func (layer *base) SetName(name string) {
	layer.name = name
}

func (layer *base) Name() string {
	if len(layer.name) == 0 {
		return layer.class
	}
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
	}, layer.derivative())
	return &ret
}

func (*base) Params() *params.Params {
	return nil
}

func (*base) Context() params.Params {
	return nil
}

func (layer *base) Print() {
	fmt.Println("  - Name:", layer.Name())
}
