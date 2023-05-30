package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type slice struct {
	a               *Tensor
	leftX, topY     int
	rightX, bottomY int
}

func (op *slice) f() *mat.Dense {
	var value mat.Dense
	value.CloneFrom(op.a.Value().Slice(op.topY, op.bottomY, op.leftX, op.rightX))
	return &value
}

func (op *slice) df(grad *Tensor) {
	rows, cols := op.a.Dims()
	delta := mat.NewDense(rows, cols, nil)
	rect := delta.Slice(op.topY, op.bottomY, op.leftX, op.rightX)
	rect.(*mat.Dense).Copy(grad.Value())
	op.a.AddGrad(delta)
	op.a.Backward(FromDense(delta))
}

func (op *slice) ZeroGrad() {
	op.a.ZeroGrad()
}

func (op *slice) needGrad() bool {
	return op.a.needGrad()
}
