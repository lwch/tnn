package tensor

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type sqrt struct {
	a      *Tensor
	value  mat.Dense
	value2 mat.Dense
}

func (op *sqrt) f() *mat.Dense {
	op.value.Apply(func(i, j int, v float64) float64 {
		return math.Sqrt(v)
	}, op.a.Value())
	op.value2.Scale(2, &op.value)
	return &op.value
}

func (op *sqrt) df(grad *Tensor) {
	var delta mat.Dense
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
