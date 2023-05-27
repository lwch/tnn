package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type maxAxis struct {
	a    *Tensor
	axis int
}

func (op *maxAxis) f() *mat.Dense {
	rows, cols := op.a.Dims()
	switch op.axis {
	case 0:
		v := mat.NewVecDense(cols, nil)
		for i := 0; i < cols; i++ {
			v.SetVec(i, mat.Max(op.a.Value().ColView(i)))
		}
		return mat.NewDense(1, cols, dupVec(v))
	case 1:
		v := mat.NewVecDense(rows, nil)
		for i := 0; i < rows; i++ {
			v.SetVec(i, mat.Max(op.a.Value().RowView(i)))
		}
		return mat.NewDense(rows, 1, dupVec(v))
	default:
		panic("invalid axis")
	}
}

func (op *maxAxis) df(grad *Tensor) {
	rows, cols := op.a.Dims()
	delta := mat.NewDense(rows, cols, nil)
	switch op.axis {
	case 0:
		v := grad.Value().RowView(0)
		for i := 0; i < rows; i++ {
			delta.RowView(i).(*mat.VecDense).CopyVec(v)
		}
	case 1:
		v := grad.Value().ColView(0)
		for i := 0; i < cols; i++ {
			delta.ColView(i).(*mat.VecDense).CopyVec(v)
		}
	default:
		panic("invalid axis")
	}
	op.a.AddGrad(delta)
	op.a.Backward(FromDense(delta))
}

func (op *maxAxis) ZeroGrad() {
	op.a.ZeroGrad()
}
