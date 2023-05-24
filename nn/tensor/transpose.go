package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type transpose struct {
	a *Tensor
}

func (op *transpose) Forward() *Tensor {
	var value mat.Dense
	value.CloneFrom(op.a.Value().T())
	return FromDense(&value)
}

func (op *transpose) Backward(grad *Tensor) {
	var delta mat.Dense
	delta.CloneFrom(grad.Value().T())
	op.a.AddGrad(&delta)
	op.a.Backward(FromDense(&delta))
}

func (op *transpose) Dims() (int, int) {
	return op.a.Dims()
}

func (op *transpose) ZeroGrad() {
	op.a.ZeroGrad()
}
