package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type add struct {
	a, b *Tensor
}

func (op *add) Forward() *Tensor {
	var value mat.Dense
	value.Add(op.a.Value(), op.b.Value())
	return FromDense(&value)
}

func (op *add) Backward(grad *Tensor) []*Tensor {
	return []*Tensor{
		grad.Clone(),
		grad.Clone(),
	}
}

func (op *add) Dims() (int, int) {
	return op.a.Dims()
}
