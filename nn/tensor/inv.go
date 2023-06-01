package tensor

import (
	"math"

	"github.com/lwch/gonum/mat32"
)

type inv struct {
	a *Tensor
}

func (op *inv) f() *mat32.Dense {
	var value mat32.Dense
	value.Apply(func(i, j int, v float32) float32 {
		return 1 / v
	}, op.a.Value())
	return &value
}

func powDense(x *mat32.Dense, n float32) *mat32.Dense {
	var value mat32.Dense
	value.Apply(func(i, j int, v float32) float32 {
		return float32(math.Pow(float64(v), float64(n)))
	}, x)
	return &value
}

func (op *inv) df(grad *Tensor) {
	var delta mat32.Dense
	delta.DivElem(grad.Value(), powDense(op.a.Value(), 2))
	delta.Scale(-1, &delta)
	op.a.AddGrad(&delta)
	op.a.Backward(FromDense(&delta))
}

func (op *inv) ZeroGrad() {
	op.a.ZeroGrad()
}

func (op *inv) needGrad() bool {
	return op.a.needGrad()
}
