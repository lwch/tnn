package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type mul struct {
	a, b *Tensor
}

func (op *mul) Forward() *Tensor {
	var value mat.Dense
	value.Mul(op.a.Value(), op.b.Value())
	return FromDense(&value)
}

func (op *mul) Backward(grad *Tensor) {
	var da, db mat.Dense
	da.Mul(grad.Value(), op.b.Value().T())
	db.Mul(op.a.Value().T(), grad.Value())
	op.a.AddGrad(&da)
	op.b.AddGrad(&db)
	op.a.Backward(FromDense(&da))
	op.b.Backward(FromDense(&db))
}

func (op *mul) Dims() (int, int) {
	rows, _ := op.a.Value().Dims()
	_, cols := op.b.Value().Dims()
	return rows, cols
}

func (op *mul) ZeroGrad() {
	op.a.ZeroGrad()
	op.b.ZeroGrad()
}

type mulElem struct {
	a, b *Tensor
}

func (op *mulElem) Forward() *Tensor {
	var value mat.Dense
	value.MulElem(op.a.Value(), op.b.Value())
	return FromDense(&value)
}

func (op *mulElem) Backward(grad *Tensor) {
	var da, db mat.Dense
	da.MulElem(op.b.Value(), grad.Value())
	db.MulElem(op.a.Value(), grad.Value())
	op.a.AddGrad(&da)
	op.b.AddGrad(&db)
	op.a.Backward(FromDense(&da))
	op.b.Backward(FromDense(&db))
}

func (op *mulElem) Dims() (int, int) {
	return op.a.Value().Dims()
}

func (op *mulElem) ZeroGrad() {
	op.a.ZeroGrad()
	op.b.ZeroGrad()
}
