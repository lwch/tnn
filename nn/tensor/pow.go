package tensor

import (
	"math"

	"github.com/lwch/gonum/mat32"
)

type pow struct {
	a         *Tensor
	b         float32
	value     mat32.Dense
	gradValue mat32.Dense
}

func (op *pow) f() *mat32.Dense {
	op.value.Apply(func(i, j int, v float32) float32 {
		return float32(math.Pow(float64(v), float64(op.b)))
	}, op.a.Value())
	var delta mat32.Dense
	delta.Apply(func(i, j int, v float32) float32 {
		return float32(math.Pow(float64(v), float64(op.b-1)))
	}, op.a.Value())
	op.gradValue.Scale(op.b, &delta)
	return &op.value
}

func (op *pow) df(grad *Tensor) {
	if !op.a.needGrad() {
		return
	}
	var delta mat32.Dense
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
