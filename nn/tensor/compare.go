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
}

func (op *maxAxis) ZeroGrad() {
	op.a.ZeroGrad()
}
