package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type mul struct {
	a, b *Tensor
}

func (op *mul) f() *mat.Dense {
	var value mat.Dense
	value.Mul(op.a.Value(), op.b.Value())
	return &value
}

func (op *mul) df(grad *Tensor) {
	if op.a.needGrad() {
		var da mat.Dense
		da.Mul(grad.Value(), op.b.Value().T())
		op.a.AddGrad(&da)
		op.a.Backward(FromDense(&da))
	}
	if op.b.needGrad() {
		var db mat.Dense
		db.Mul(op.a.Value().T(), grad.Value())
		op.b.AddGrad(&db)
		op.b.Backward(FromDense(&db))
	}
}

func (op *mul) ZeroGrad() {
	op.a.ZeroGrad()
	op.b.ZeroGrad()
}

func (op *mul) needGrad() bool {
	if op.a.needGrad() {
		return true
	}
	return op.b.needGrad()
}

type mulElem struct {
	a, b *Tensor
}

func (op *mulElem) f() *mat.Dense {
	var value mat.Dense
	value.MulElem(op.a.Value(), op.b.Value())
	return &value
}

func (op *mulElem) df(grad *Tensor) {
	if op.a.needGrad() {
		var da mat.Dense
		da.MulElem(op.b.Value(), grad.Value())
		op.a.AddGrad(&da)
		op.a.Backward(FromDense(&da))
	}
	if op.b.needGrad() {
		var db mat.Dense
		db.MulElem(op.a.Value(), grad.Value())
		op.b.AddGrad(&db)
		op.b.Backward(FromDense(&db))
	}
}

func (op *mulElem) ZeroGrad() {
	op.a.ZeroGrad()
	op.b.ZeroGrad()
}

func (op *mulElem) needGrad() bool {
	if op.a.needGrad() {
		return true
	}
	return op.b.needGrad()
}
