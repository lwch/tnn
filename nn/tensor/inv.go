package tensor

import (
	"math"

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

func powDense(x *mat.Dense, n float64) *mat.Dense {
	var value mat.Dense
	value.Apply(func(i, j int, v float64) float64 {
		return math.Pow(v, n)
	}, x)
	return &value
}

func (op *inv) Backward(grad *Tensor) {
	var delta mat.Dense
	delta.DivElem(grad.Value(), powDense(op.a.Value(), 2))
	delta.Scale(-1, &delta)
	if op.a.grad == nil {
		op.a.grad = Zeros(delta.Dims())
	}
	op.a.grad.AddValue(&delta)
	op.a.Backward(FromDense(&delta))
}

func (op *inv) Dims() (int, int) {
	return op.a.Dims()
}

func (op *inv) ZeroGrad() {
	op.a.ZeroGrad()
}
