package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type slice struct {
	a               *Tensor
	leftX, topY     int
	rightX, bottomY int
}

func (op *slice) Forward() *Tensor {
	var value mat.Dense
	value.CloneFrom(op.a.Value().Slice(op.topY, op.bottomY, op.leftX, op.rightX))
	return FromDense(&value)
}

func (op *slice) Backward(grad *Tensor) {
	rows, cols := op.a.Dims()
	delta := mat.NewDense(rows, cols, nil)
	rect := delta.Slice(op.topY, op.bottomY, op.leftX, op.rightX)
	rect.(*mat.Dense).Copy(grad.Value())
	op.a.AddGrad(delta)
	op.a.Backward(FromDense(delta))
}

func (op *slice) Dims() (int, int) {
	return op.bottomY - op.topY, op.rightX - op.leftX
}

func (op *slice) ZeroGrad() {
	op.a.ZeroGrad()
}
