package tensor

import (
	"math"

	"github.com/lwch/gonum/mat32"
)

type sqrt struct {
	a      *Tensor
	value  mat32.Dense
	value2 mat32.Dense
}

func (op *sqrt) f() *mat32.Dense {
	op.value.Apply(func(i, j int, v float32) float32 {
		return float32(math.Sqrt(float64(v)))
	}, op.a.Value())
	op.value2.Scale(2, &op.value)
	return &op.value
}

func (op *sqrt) df(grad *Tensor) {
	if !op.a.needGrad() {
		return
	}
	var delta mat32.Dense
	delta.DivElem(grad.Value(), &op.value2)
	op.a.AddGrad(&delta)
	op.a.Backward(FromDense(&delta))
}

func (op *sqrt) ZeroGrad() {
	op.a.ZeroGrad()
}

func (op *sqrt) needGrad() bool {
	return op.a.needGrad()
}
