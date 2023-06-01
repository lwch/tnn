package tensor

import (
	"math"

	"github.com/lwch/gonum/mat32"
)

type exp struct {
	a     *Tensor
	value mat32.Dense
}

func (op *exp) f() *mat32.Dense {
	op.value.Apply(func(i, j int, v float32) float32 {
		return float32(math.Exp(float64(v)))
	}, op.a.Value())
	return &op.value
}

func (op *exp) df(grad *Tensor) {
	if !op.a.needGrad() {
		return
	}
	var delta mat32.Dense
	delta.MulElem(grad.Value(), &op.value)
	op.a.AddGrad(&delta)
	op.a.Backward(FromDense(&delta))
}

func (op *exp) ZeroGrad() {
	op.a.ZeroGrad()
}

func (op *exp) needGrad() bool {
	return op.a.needGrad()
}
