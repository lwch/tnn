package tensor

import "gonum.org/v1/gonum/mat"

type sub struct {
	a, b *Tensor
}

func (op *sub) Forward() *Tensor {
	var value mat.Dense
	value.Sub(op.a.Value(), op.b.Value())
	return FromDense(&value)
}

func (op *sub) Backward(grad *Tensor) []*Tensor {
	return []*Tensor{
		grad.Clone(),
		grad.Clone().Negate(),
	}
}

func (op *sub) Dims() (int, int) {
	return op.a.Dims()
}
