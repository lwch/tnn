package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type conact struct {
	a *Tensor
	b *Tensor
}

func (op *conact) f() *mat.Dense {
	var value mat.Dense
	value.Augment(op.a.Value(), op.b.Value())
	return &value
}

func (op *conact) df(grad *Tensor) {
	aRows, aCols := op.a.Dims()
	bRows, bCols := op.b.Dims()
	if op.a.needGrad() {
		da := grad.Value().Slice(0, aRows, 0, aCols)
		op.a.AddGrad(da.(*mat.Dense))
		op.a.Backward(FromDense(da.(*mat.Dense)))
	}
	if op.b.needGrad() {
		db := grad.Value().Slice(0, bRows, aCols, aCols+bCols)
		op.b.AddGrad(db.(*mat.Dense))
		op.b.Backward(FromDense(db.(*mat.Dense)))
	}
}

func (op *conact) ZeroGrad() {
	op.a.ZeroGrad()
}

func (op *conact) needGrad() bool {
	if op.a.needGrad() {
		return true
	}
	return op.b.needGrad()
}

type stack struct {
	a *Tensor
	b *Tensor
}

func (op *stack) f() *mat.Dense {
	var stack mat.Dense
	stack.Stack(op.a.Value(), op.b.Value())
	return &stack
}

func (op *stack) df(grad *Tensor) {
	aRows, aCols := op.a.Dims()
	bRows, bCols := op.b.Dims()
	if op.a.needGrad() {
		da := grad.Value().Slice(0, aRows, 0, aCols)
		op.a.AddGrad(da.(*mat.Dense))
		op.a.Backward(FromDense(da.(*mat.Dense)))
	}
	if op.b.needGrad() {
		db := grad.Value().Slice(aRows, aRows+bRows, 0, bCols)
		op.b.AddGrad(db.(*mat.Dense))
		op.b.Backward(FromDense(db.(*mat.Dense)))
	}
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
