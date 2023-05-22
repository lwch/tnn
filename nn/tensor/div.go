package tensor

import "gonum.org/v1/gonum/mat"

type divElem struct {
	a, b *Tensor
}

func (op *divElem) Forward() *Tensor {
	var value mat.Dense
	value.DivElem(op.a.Value(), op.b.Value())
	return FromDense(&value)
}

func (op *divElem) Backward(grad *Tensor) {
	var da mat.Dense
	da.DivElem(grad.Value(), op.b.Value())
	var db mat.Dense
	db.Scale(-1, grad.Value())
	db.MulElem(&db, op.a.Value())
	db.DivElem(&db, powDense(op.b.Value(), 2))
	if op.a.grad == nil {
		op.a.grad = Zeros(da.Dims())
	}
	if op.b.grad == nil {
		op.b.grad = Zeros(db.Dims())
	}
	op.a.grad.AddValue(&da)
	op.b.grad.AddValue(&db)
	op.a.Backward(FromDense(&da))
	op.b.Backward(FromDense(&db))
}

func (op *divElem) Dims() (int, int) {
	return op.a.Value().Dims()
}

func (op *divElem) ZeroGrad() {
	op.a.ZeroGrad()
	op.b.ZeroGrad()
}
