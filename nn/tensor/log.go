package tensor

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type log struct {
	a *Tensor
}

func (op *log) logValue() *mat.Dense {
	var value mat.Dense
	value.Apply(func(i, j int, v float64) float64 {
		return math.Log(v)
	}, op.a.Value())
	return &value
}

func (op *log) Forward() *Tensor {
	return FromDense(op.logValue())
}

func (op *log) Backward(grad *Tensor) {
	var delta mat.Dense
	delta.Apply(func(i, j int, v float64) float64 {
		return 1 / v
	}, op.a.Value())
	delta.MulElem(grad.Value(), &delta)
	op.a.grad = FromDense(&delta)
	op.a.Backward(op.a.grad)
}

func (op *log) Dims() (int, int) {
	return op.a.Dims()
}
