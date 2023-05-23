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

func (op *tanh) Forward() *Tensor {
	return FromDense(tanhDense(op.a.Value()))
}

func (op *tanh) Backward(grad *Tensor) {
	a := Ones(grad.Dims())
	b := powDense(tanhDense(grad.Value()), 2)
	var delta mat.Dense
	delta.Sub(a.Value(), b)
	if op.a.grad == nil {
		op.a.grad = Zeros(grad.Dims())
	}
	op.a.grad.AddValue(&delta)
	op.a.Backward(FromDense(&delta))
}

func (op *tanh) Dims() (int, int) {
	return op.a.Dims()
}

func (op *tanh) ZeroGrad() {
	op.a.ZeroGrad()
}
