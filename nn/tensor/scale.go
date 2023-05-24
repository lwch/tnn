package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type scale struct {
	a float64
	b *Tensor
}

func (op *scale) Forward() *Tensor {
	var value mat.Dense
	value.Scale(op.a, op.b.Value())
	return FromDense(&value)
}

func (op *scale) Backward(grad *Tensor) {
	var delta mat.Dense
	delta.Scale(op.a, grad.Value())
	op.b.AddGrad(&delta)
	op.b.Backward(FromDense(&delta))
}

func (op *scale) Dims() (int, int) {
	return op.b.Dims()
}

func (op *scale) ZeroGrad() {
	op.b.ZeroGrad()
}
