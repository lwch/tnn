package tensor

import (
	"math"

	"github.com/lwch/gonum/mat32"
)

type log struct {
	a   *Tensor
	log mat32.Dense
	inv mat32.Dense
}

func (op *log) f() *mat32.Dense {
	op.log.Apply(func(i, j int, v float32) float32 {
		return float32(math.Log(float64(v)))
	}, op.a.Value())
	op.inv.Apply(func(i, j int, v float32) float32 {
		return 1 / v
	}, op.a.Value())
	return &op.log
}

func (op *log) df(grad *Tensor) {
	if !op.a.needGrad() {
		return
	}
	var delta mat32.Dense
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
