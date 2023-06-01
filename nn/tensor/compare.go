package tensor

import "github.com/lwch/gonum/mat32"

type maxAxis struct {
	a    *Tensor
	axis int
}

func (op *maxAxis) f() *mat32.Dense {
	rows, cols := op.a.Dims()
	switch op.axis {
	case 0:
		v := mat32.NewVecDense(cols, nil)
		for i := 0; i < cols; i++ {
			v.SetVec(i, mat32.Max(op.a.Value().ColView(i)))
		}
		return mat32.NewDense(1, cols, dupVec(v))
	case 1:
		v := mat32.NewVecDense(rows, nil)
		for i := 0; i < rows; i++ {
			v.SetVec(i, mat32.Max(op.a.Value().RowView(i)))
		}
		return mat32.NewDense(rows, 1, dupVec(v))
	default:
		panic("invalid axis")
	}
}

func (op *maxAxis) df(grad *Tensor) {
	if !op.a.needGrad() {
		return
	}
	rows, cols := op.a.Dims()
	delta := mat32.NewDense(rows, cols, nil)
	switch op.axis {
	case 0:
		v := grad.Value().RowView(0)
		for i := 0; i < rows; i++ {
			delta.RowView(i).(*mat32.VecDense).CopyVec(v)
		}
	case 1:
		v := grad.Value().ColView(0)
		for i := 0; i < cols; i++ {
			delta.ColView(i).(*mat32.VecDense).CopyVec(v)
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

func (op *maxAxis) needGrad() bool {
	return op.a.needGrad()
}
