package tensor

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type pow struct {
	a         *Tensor
	b         float64
	value     mat.Dense
	gradValue mat.Dense
}

func (op *pow) f() *mat.Dense {
	op.value.Apply(func(i, j int, v float64) float64 {
		return math.Pow(v, op.b)
	}, op.a.Value())
	var delta mat.Dense
	delta.Apply(func(i, j int, v float64) float64 {
		return math.Pow(v, op.b-1)
	}, op.a.Value())
	op.gradValue.Scale(op.b, &delta)
	return &op.value
}

func (op *pow) df(grad *Tensor) {
	if !op.a.needGrad() {
		return
	}
	var delta mat.Dense
	delta.MulElem(grad.Value(), &op.gradValue)
	op.a.AddGrad(&delta)
	op.a.Backward(FromDense(&delta))
}

func (op *pow) ZeroGrad() {
	op.a.ZeroGrad()
}

func (op *pow) needGrad() bool {
	return op.a.needGrad()
}
