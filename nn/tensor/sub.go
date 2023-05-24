package tensor

import "gonum.org/v1/gonum/mat"

type sub struct {
	a, b *Tensor
}

func (op *sub) Forward() *Tensor {
	var value mat.Dense
	value.Sub(op.a.Value(), op.b.Value())
	return FromDense(&value)
}

func (op *sub) Backward(grad *Tensor) {
	da := grad.Clone()
	db := grad.Scale(-1)
	op.a.AddGrad(da.Value())
	op.b.AddGrad(db.Value())
	op.a.Backward(da)
	op.b.Backward(db)
}

func (op *sub) Dims() (int, int) {
	return op.a.Dims()
}

func (op *sub) ZeroGrad() {
	op.a.ZeroGrad()
	op.b.ZeroGrad()
}
