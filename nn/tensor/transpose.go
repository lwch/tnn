package tensor

import "github.com/lwch/gonum/mat32"

type transpose struct {
	a *Tensor
}

func (op *transpose) f() *mat32.Dense {
	var value mat32.Dense
	value.CloneFrom(op.a.Value().T())
	return &value
}

func (op *transpose) df(grad *Tensor) {
	if !op.a.needGrad() {
		return
	}
	var delta mat32.Dense
	delta.CloneFrom(grad.Value().T())
	op.a.AddGrad(&delta)
	op.a.Backward(FromDense(&delta))
}

func (op *transpose) ZeroGrad() {
	op.a.ZeroGrad()
}

func (op *transpose) needGrad() bool {
	return op.a.needGrad()
}
