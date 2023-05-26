package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type divElem struct {
	a, b *Tensor
}

func (op *divElem) f() *mat.Dense {
	var value mat.Dense
	value.DivElem(op.a.Value(), op.b.Value())
	return &value
}

func (op *divElem) df(grad *Tensor) {
	var da mat.Dense
	da.DivElem(grad.Value(), op.b.Value())
	var db mat.Dense
	db.Scale(-1, grad.Value())
	db.MulElem(&db, op.a.Value())
	db.DivElem(&db, powDense(op.b.Value(), 2))
	op.a.AddGrad(&da)
	op.b.AddGrad(&db)
	op.a.Backward(FromDense(&da))
	op.b.Backward(FromDense(&db))
}

func (op *divElem) ZeroGrad() {
	op.a.ZeroGrad()
	op.b.ZeroGrad()
}
