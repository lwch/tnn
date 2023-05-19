package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type mul struct {
	a *Tensor
	b *Tensor
}

func (op *mul) Forward() *Tensor {
	var value mat.Dense
	value.Mul(op.a.Value(), op.b.Value())
	return FromDense(&value)
}

func (op *mul) Backward(grad *Tensor) []*Tensor {
	var da, db mat.Dense
	da.Mul(op.b.Value(), grad.Value())
	db.Mul(op.a.Value().T(), grad.Value())
	return []*Tensor{
		FromDense(mat.DenseCopyOf(da.T())),
		FromDense(&db),
	}
}

func (op *mul) Dims() (int, int) {
	rows, _ := op.a.Value().Dims()
	_, cols := op.b.Value().Dims()
	return rows, cols
}
