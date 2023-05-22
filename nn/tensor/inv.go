package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type inv struct {
	a *Tensor
}

func (op *inv) Forward() *Tensor {
	var value mat.Dense
	value.Apply(func(i, j int, v float64) float64 {
		return 1 / v
	}, op.a.Value())
	return FromDense(&value)
}

func (op *inv) Backward(grad *Tensor) {
	var pow mat.Dense
	pow.Pow(op.a.Value(), 2)
	var delta mat.Dense
	delta.DivElem(grad.Value(), &pow)
	op.a.grad = FromDense(&delta)
	op.a.Backward(op.a.grad)
}

func (op *inv) Dims() (int, int) {
	return op.a.Dims()
}
