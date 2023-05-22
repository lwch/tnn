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
	var db mat.Dense
	db.Scale(op.a, grad.Value())
	op.b.grad = FromDense(&db)
	op.b.Backward(op.b.grad)
}

func (op *scale) Dims() (int, int) {
	return op.b.Dims()
}
