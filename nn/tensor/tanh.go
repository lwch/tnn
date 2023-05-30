package tensor

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type tanh struct {
	a *Tensor
}

func tanhDense(x *mat.Dense) *mat.Dense {
	var value mat.Dense
	value.Apply(func(i, j int, v float64) float64 {
		return math.Tanh(v)
	}, x)
	return &value
}

func (op *tanh) f() *mat.Dense {
	return tanhDense(op.a.Value())
}

func (op *tanh) df(grad *Tensor) {
	a := Ones(grad.Dims())
	b := powDense(tanhDense(grad.Value()), 2)
	var delta mat.Dense
	delta.Sub(a.Value(), b)
	op.a.AddGrad(&delta)
	op.a.Backward(FromDense(&delta))
}

func (op *tanh) ZeroGrad() {
	op.a.ZeroGrad()
}

func (op *tanh) needGrad() bool {
	return op.a.needGrad()
}
