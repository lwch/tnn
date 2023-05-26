package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type sum struct {
	a *Tensor
}

func (op *sum) Forward() *Tensor {
	n := mat.Sum(op.a.Value())
	return New([]float64{n}, 1, 1)
}

func (op *sum) Backward(grad *Tensor) {
	rows, cols := op.a.Value().Dims()
	delta := Numbers(rows, cols, grad.Value().At(0, 0))
	op.a.AddGrad(delta.Value())
	op.a.Backward(delta)
}

func (op *sum) Dims() (int, int) {
	return op.a.Dims()
}

func (op *sum) ZeroGrad() {
	op.a.ZeroGrad()
}

type sumAxis struct {
	a    *Tensor
	axis int
}

func (op *sumAxis) Forward() *Tensor {
	rows, cols := op.a.Dims()
	switch op.axis {
	case 0:
		v := mat.NewVecDense(cols, nil)
		for i := 0; i < rows; i++ {
			v.AddVec(v, op.a.Value().RowView(i))
		}
		return FromRowVector(v)
	case 1:
		v := mat.NewVecDense(rows, nil)
		for i := 0; i < cols; i++ {
			v.AddVec(v, op.a.Value().ColView(i))
		}
		return FromColVector(v)
	default:
		panic("invalid axis")
	}
}

func (op *sumAxis) Backward(grad *Tensor) {
	rows, cols := op.a.Value().Dims()
	delta := mat.NewDense(rows, cols, nil)
	switch op.axis {
	case 0:
		for i := 0; i < rows; i++ {
			delta.SetRow(i, dupVec(grad.Value().RowView(0).(*mat.VecDense)))
		}
	case 1:
		for i := 0; i < cols; i++ {
			delta.SetCol(i, dupVec(grad.Value().ColView(0).(*mat.VecDense)))
		}
	default:
		panic("invalid axis")
	}
	op.a.AddGrad(delta)
	op.a.Backward(FromDense(delta))
}

func (op *sumAxis) Dims() (int, int) {
	rows, cols := op.a.Dims()
	switch op.axis {
	case 0:
		return 1, cols
	case 1:
		return rows, 1
	default:
		panic("invalid axis")
	}
}

func (op *sumAxis) ZeroGrad() {
	op.a.ZeroGrad()
}
