package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type transpose struct {
	a *Tensor
}

func (op *transpose) f() *mat.Dense {
	var value mat.Dense
	value.CloneFrom(op.a.Value().T())
	return &value
}

func (op *transpose) df(grad *Tensor) {
	var delta mat.Dense
	delta.CloneFrom(grad.Value().T())
	op.a.AddGrad(&delta)
	op.a.Backward(FromDense(&delta))
}

func (op *transpose) ZeroGrad() {
	op.a.ZeroGrad()
}
