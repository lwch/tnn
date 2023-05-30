package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type meanAxis struct {
	a    *Tensor
	axis int
}

func (op *meanAxis) f() *mat.Dense {
	rows, cols := op.a.Dims()
	switch op.axis {
	case 0:
		value := mat.NewDense(1, cols, nil)
		for i := 0; i < cols; i++ {
			value.Set(0, i, mat.Sum(op.a.Value().ColView(i))/float64(rows))
		}
		return value
	case 1:
		value := mat.NewDense(rows, 1, nil)
		for i := 0; i < rows; i++ {
			value.Set(i, 0, mat.Sum(op.a.Value().RowView(i))/float64(cols))
		}
		return value
	default:
		panic("invalid axis")
	}
}

func (op *meanAxis) df(grad *Tensor) {
	rows, cols := op.a.Dims()
	delta := mat.NewDense(rows, cols, nil)
	switch op.axis {
	case 0:
		for i := 0; i < cols; i++ {
			n := grad.Value().At(0, i) / float64(rows)
			for j := 0; j < rows; j++ {
				delta.Set(j, i, n)
			}
		}
	case 1:
		for i := 0; i < rows; i++ {
			n := grad.Value().At(i, 0) / float64(cols)
			for j := 0; j < cols; j++ {
				delta.Set(i, j, n)
			}
		}
	default:
		panic("invalid axis")
	}
	op.a.AddGrad(delta)
	op.a.Backward(FromDense(delta))
}

func (op *meanAxis) ZeroGrad() {
	op.a.ZeroGrad()
}

func (op *meanAxis) needGrad() bool {
	return op.a.needGrad()
}
