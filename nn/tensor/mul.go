package tensor

import "github.com/lwch/gonum/mat32"

type mul struct {
	a, b *Tensor
}

func (op *mul) f() *mat32.Dense {
	var value mat32.Dense
	value.Mul(op.a.Value(), op.b.Value())
	return &value
}

func (op *mul) df(grad *Tensor) {
	if op.a.needGrad() {
		var da mat32.Dense
		da.Mul(grad.Value(), op.b.Value().T())
		op.a.AddGrad(&da)
		op.a.Backward(FromDense(&da))
	}
	if op.b.needGrad() {
		var db mat32.Dense
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

func (op *mulElem) f() *mat32.Dense {
	var value mat32.Dense
	value.MulElem(op.a.Value(), op.b.Value())
	return &value
}

func (op *mulElem) df(grad *Tensor) {
	if op.a.needGrad() {
		var da mat32.Dense
		da.MulElem(op.b.Value(), grad.Value())
		op.a.AddGrad(&da)
		op.a.Backward(FromDense(&da))
	}
	if op.b.needGrad() {
		var db mat32.Dense
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
