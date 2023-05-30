package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type stack struct {
	a *Tensor
	b *Tensor
}

func (op *stack) f() *mat.Dense {
	var stack mat.Dense
	stack.Stack(op.a.Value().T(), op.b.Value().T())
	var value mat.Dense
	value.CloneFrom(stack.T())
	return &value
}

func (op *stack) df(grad *Tensor) {
	aRows, aCols := op.a.Dims()
	bRows, bCols := op.b.Dims()
	da := grad.Value().Slice(0, aRows, 0, aCols)
	db := grad.Value().Slice(0, bRows, aCols, aCols+bCols)
	op.a.AddGrad(da.(*mat.Dense))
	op.b.AddGrad(db.(*mat.Dense))
	op.a.Backward(FromDense(da.(*mat.Dense)))
	op.b.Backward(FromDense(db.(*mat.Dense)))
}

func (op *stack) ZeroGrad() {
	op.a.ZeroGrad()
}

func (op *stack) needGrad() bool {
	if op.a.needGrad() {
		return true
	}
	return op.b.needGrad()
}
