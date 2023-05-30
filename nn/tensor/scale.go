package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type scale struct {
	a float64
	b *Tensor
}

func (op *scale) f() *mat.Dense {
	var value mat.Dense
	value.Scale(op.a, op.b.Value())
	return &value
}

func (op *scale) df(grad *Tensor) {
	if !op.b.needGrad() {
		return
	}
	var delta mat.Dense
	delta.Scale(op.a, grad.Value())
	op.b.AddGrad(&delta)
	op.b.Backward(FromDense(&delta))
}

func (op *scale) ZeroGrad() {
	op.b.ZeroGrad()
}

func (op *scale) needGrad() bool {
	return op.b.needGrad()
}
