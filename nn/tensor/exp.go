package tensor

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type exp struct {
	a *Tensor
}

func (op *exp) expValue() *mat.Dense {
	var value mat.Dense
	value.Apply(func(i, j int, v float64) float64 {
		return math.Exp(v)
	}, op.a.Value())
	return &value
}

func (op *exp) Forward() *Tensor {
	return FromDense(op.expValue())
}

func (op *exp) Backward(grad *Tensor) {
	delta := op.expValue()
	delta.MulElem(grad.Value(), delta)
	if op.a.grad == nil {
		op.a.grad = Zeros(delta.Dims())
	}
	op.a.grad.AddValue(delta)
	op.a.Backward(FromDense(delta))
}

func (op *exp) Dims() (int, int) {
	return op.a.Dims()
}

func (op *exp) ZeroGrad() {
	op.a.ZeroGrad()
}
