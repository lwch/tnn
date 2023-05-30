package tensor

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type exp struct {
	a     *Tensor
	value mat.Dense
}

func (op *exp) f() *mat.Dense {
	op.value.Apply(func(i, j int, v float64) float64 {
		return math.Exp(v)
	}, op.a.Value())
	return &op.value
}

func (op *exp) df(grad *Tensor) {
	var delta mat.Dense
	delta.MulElem(grad.Value(), &op.value)
	op.a.AddGrad(&delta)
	op.a.Backward(FromDense(&delta))
}

func (op *exp) ZeroGrad() {
	op.a.ZeroGrad()
}

func (op *exp) needGrad() bool {
	return op.a.needGrad()
}
