package tensor

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type log struct {
	a   *Tensor
	log mat.Dense
	inv mat.Dense
}

func (op *log) f() *mat.Dense {
	op.log.Apply(func(i, j int, v float64) float64 {
		return math.Log(v)
	}, op.a.Value())
	op.inv.Apply(func(i, j int, v float64) float64 {
		return 1 / v
	}, op.a.Value())
	return &op.log
}

func (op *log) df(grad *Tensor) {
	if !op.a.needGrad() {
		return
	}
	var delta mat.Dense
	delta.MulElem(grad.Value(), &op.inv)
	op.a.AddGrad(&delta)
	op.a.Backward(FromDense(&delta))
}

func (op *log) ZeroGrad() {
	op.a.ZeroGrad()
}

func (op *log) needGrad() bool {
	return op.a.needGrad()
}
