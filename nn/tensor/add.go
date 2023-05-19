package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type add struct {
	a, b *Tensor
}

func (op *add) Forward() *Tensor {
	var value mat.VecDense
	value.AddVec(op.a.Value(), op.b.Value())
	return FromVec(&value, op.a.Shapes()...)
}

func (op *add) Backward(grad *Tensor) []*Tensor {
	if grad == nil {
		grad = Ones(op.a.Shapes()...)
	}
	return []*Tensor{
		grad.Clone(),
		grad.Clone(),
	}
}
