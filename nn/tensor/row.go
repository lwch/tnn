package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type row2Matrix struct {
	a    *Tensor
	n    int // 获取行号
	rows int // 变成多少行的matrix
	cols int // 变成多少列的matrix
}

func (op *row2Matrix) f() *mat.Dense {
	_, cols := op.a.Dims()
	if cols%op.cols != 0 {
		panic("cols %% op.cols != 0")
	}
	value := mat.NewDense(op.rows, op.cols, nil)
	row := op.a.Value().RowView(op.n)
	for i := 0; i < cols/op.cols; i++ {
		start := i * op.cols
		col := row.(*mat.VecDense).SliceVec(start, start+op.cols)
		value.SetRow(i, dupVec(col.(*mat.VecDense)))
	}
	return value
}

func (op *row2Matrix) df(grad *Tensor) {
	if !op.a.needGrad() {
		return
	}
	rows, cols := op.a.Dims()
	delta := mat.NewDense(rows, cols, nil)
	row := delta.RowView(op.n).(*mat.VecDense)
	rows, _ = grad.Dims()
	for i := 0; i < rows; i++ {
		start := i * op.cols
		to := row.SliceVec(start, start+op.cols).(*mat.VecDense)
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

func (op *rowVector) f() *mat.Dense {
	rows, cols := op.a.Dims()
	value := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		start := i * cols
		copy(value[start:start+cols], dupVec(op.a.Value().RowView(i).(*mat.VecDense)))
	}
	return mat.NewDense(1, rows*cols, value)
}

func (op *rowVector) df(grad *Tensor) {
	if !op.a.needGrad() {
		return
	}
	rows, cols := op.a.Dims()
	delta := mat.NewDense(rows, cols, nil)
	row := grad.Value().RowView(0).(*mat.VecDense)
	for i := 0; i < rows; i++ {
		start := i * cols
		src := row.SliceVec(start, start+cols)
		delta.RowView(i).(*mat.VecDense).CopyVec(src)
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

type appendRow struct {
	a *Tensor
	b *Tensor
}

func (op *appendRow) f() *mat.Dense {
	aRows, aCols := op.a.Dims()
	bRows, bCols := op.b.Dims()
	if aCols != 0 && aCols != bCols {
		panic("aCols != bCols")
	}
	value := make([]float64, (aRows+bRows)*bCols)
	for i := 0; i < aRows; i++ {
		start := i * aCols
		row := dupVec(op.a.Value().RowView(i).(*mat.VecDense))
		copy(value[start:start+aCols], row)
	}
	for i := 0; i < bRows; i++ {
		start := (aRows + i) * bCols
		row := dupVec(op.b.Value().RowView(i).(*mat.VecDense))
		copy(value[start:start+bCols], row)
	}
	return mat.NewDense(aRows+bRows, bCols, value)
}

func (op *appendRow) op(grad *Tensor, a bool) {
	var target *Tensor
	var gradDense *mat.Dense
	var rows, cols int
	if a {
		target = op.a
		rows, cols = op.a.Dims()
		gradDense = grad.Value().Slice(0, rows, 0, cols).(*mat.Dense)
	} else {
		target = op.b
		aRows, _ := op.a.Dims()
		rows, cols = op.b.Dims()
		gradDense = grad.Value().Slice(aRows, aRows+rows, 0, cols).(*mat.Dense)
	}
	if rows == 0 {
		return
	}
	delta := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		src := gradDense.RowView(i).(*mat.VecDense)
		delta.RowView(i).(*mat.VecDense).CopyVec(src)
	}
	target.AddGrad(delta)
	target.Backward(FromDense(delta))
}

func (op *appendRow) df(grad *Tensor) {
	if op.a.needGrad() {
		op.op(grad, true)
	}
	if op.b.needGrad() {
		op.op(grad, false)
	}
}

func (op *appendRow) ZeroGrad() {
	op.a.ZeroGrad()
	op.b.ZeroGrad()
}

func (op *appendRow) needGrad() bool {
	if op.a.needGrad() {
		return true
	}
	return op.b.needGrad()
}
