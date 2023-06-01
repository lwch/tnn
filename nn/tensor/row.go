package tensor

import "github.com/lwch/gonum/mat32"

// TODO: 优化row2Matrix和rowVector的内存分配

type row2Matrix struct {
	a    *Tensor
	n    int // 获取行号
	rows int // 变成多少行的matrix
	cols int // 变成多少列的matrix
}

func (op *row2Matrix) f() *mat32.Dense {
	_, cols := op.a.Dims()
	if cols%op.cols != 0 {
		panic("cols %% op.cols != 0")
	}
	value := mat32.NewDense(op.rows, op.cols, nil)
	row := op.a.Value().RowView(op.n)
	for i := 0; i < cols/op.cols; i++ {
		start := i * op.cols
		col := row.(*mat32.VecDense).SliceVec(start, start+op.cols)
		value.SetRow(i, dupVec(col.(*mat32.VecDense)))
	}
	return value
}

func (op *row2Matrix) df(grad *Tensor) {
	if !op.a.needGrad() {
		return
	}
	rows, cols := op.a.Dims()
	delta := mat32.NewDense(rows, cols, nil)
	row := delta.RowView(op.n).(*mat32.VecDense)
	rows, _ = grad.Dims()
	for i := 0; i < rows; i++ {
		start := i * op.cols
		to := row.SliceVec(start, start+op.cols).(*mat32.VecDense)
		to.CopyVec(grad.Value().RowView(i))
	}
	op.a.AddGrad(delta)
	op.a.Backward(FromDense(delta))
}

func (op *row2Matrix) ZeroGrad() {
	op.a.ZeroGrad()
}

func (op *row2Matrix) needGrad() bool {
	return op.a.needGrad()
}

type rowVector struct {
	a *Tensor
}

func (op *rowVector) f() *mat32.Dense {
	rows, cols := op.a.Dims()
	value := make([]float32, rows*cols)
	for i := 0; i < rows; i++ {
		start := i * cols
		copy(value[start:start+cols], dupVec(op.a.Value().RowView(i).(*mat32.VecDense)))
	}
	return mat32.NewDense(1, rows*cols, value)
}

func (op *rowVector) df(grad *Tensor) {
	if !op.a.needGrad() {
		return
	}
	rows, cols := op.a.Dims()
	delta := mat32.NewDense(rows, cols, nil)
	row := grad.Value().RowView(0).(*mat32.VecDense)
	for i := 0; i < rows; i++ {
		start := i * cols
		src := row.SliceVec(start, start+cols)
		delta.RowView(i).(*mat32.VecDense).CopyVec(src)
	}
	op.a.AddGrad(delta)
	op.a.Backward(FromDense(delta))
}

func (op *rowVector) ZeroGrad() {
	op.a.ZeroGrad()
}

func (op *rowVector) needGrad() bool {
	return op.a.needGrad()
}
