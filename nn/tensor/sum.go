package tensor

import "github.com/lwch/gonum/mat32"

type sum struct {
	a *Tensor
}

func (op *sum) f() *mat32.Dense {
	n := mat32.Sum(op.a.Value())
	return mat32.NewDense(1, 1, []float32{n})
}

func (op *sum) df(grad *Tensor) {
	if !op.a.needGrad() {
		return
	}
	rows, cols := op.a.Value().Dims()
	delta := Numbers(rows, cols, grad.Value().At(0, 0))
	op.a.AddGrad(delta.Value())
	op.a.Backward(delta)
}

func (op *sum) ZeroGrad() {
	op.a.ZeroGrad()
}

func (op *sum) needGrad() bool {
	return op.a.needGrad()
}

type sumAxis struct {
	a    *Tensor
	axis int
}

func (op *sumAxis) f() *mat32.Dense {
	rows, cols := op.a.Dims()
	switch op.axis {
	case 0:
		v := mat32.NewVecDense(cols, nil)
		for i := 0; i < rows; i++ {
			v.AddVec(v, op.a.Value().RowView(i))
		}
		return mat32.NewDense(1, cols, dupVec(v))
	case 1:
		v := mat32.NewVecDense(rows, nil)
		for i := 0; i < cols; i++ {
			v.AddVec(v, op.a.Value().ColView(i))
		}
		return mat32.NewDense(rows, 1, dupVec(v))
	default:
		panic("invalid axis")
	}
}

func (op *sumAxis) df(grad *Tensor) {
	if !op.a.needGrad() {
		return
	}
	rows, cols := op.a.Value().Dims()
	delta := mat32.NewDense(rows, cols, nil)
	switch op.axis {
	case 0:
		row := dupVec(grad.Value().RowView(0).(*mat32.VecDense))
		for i := 0; i < rows; i++ {
			delta.SetRow(i, row)
		}
	case 1:
		col := dupVec(grad.Value().ColView(0).(*mat32.VecDense))
		for i := 0; i < cols; i++ {
			delta.SetCol(i, col)
		}
	default:
		panic("invalid axis")
	}
	op.a.AddGrad(delta)
	op.a.Backward(FromDense(delta))
}

func (op *sumAxis) ZeroGrad() {
	op.a.ZeroGrad()
}

func (op *sumAxis) needGrad() bool {
	return op.a.needGrad()
}
