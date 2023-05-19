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

func (op *scale) Backward(grad *Tensor) []*Tensor {
	var da, db mat.Dense
	da.Scale(op.a, grad.Value())
	db.MulElem(op.b.Value(), grad.Value())
	return []*Tensor{
		FromDense(&da),
		FromDense(&db),
	}
}

func (op *scale) Dims() (int, int) {
	return op.b.Dims()
}
