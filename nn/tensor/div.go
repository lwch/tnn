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
	var pow mat.Dense
	pow.Pow(op.b.Value(), 2)
	var db mat.Dense
	db.Scale(-1, grad.Value())
	db.MulElem(&db, op.a.Value())
	db.DivElem(&db, &pow)
	op.a.grad = FromDense(&da)
	op.b.grad = FromDense(&db)
	op.a.Backward(op.a.grad)
	op.b.Backward(op.b.grad)
}

func (op *divElem) Dims() (int, int) {
	return op.a.Value().Dims()
}
