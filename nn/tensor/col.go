package tensor

import "gonum.org/v1/gonum/mat"

type appendCol struct {
	a *Tensor
	b *Tensor
}

func (op *appendCol) f() *mat.Dense {
	aRows, aCols := op.a.Dims()
	bRows, bCols := op.b.Dims()
	if aCols != 0 && aRows != bRows {
		panic("aRows != bRows")
	}
	value := make([]float64, bRows*(aCols+bCols))
	for i := 0; i < aRows; i++ {
		start := i * (aCols + bCols)
		row := dupVec(op.a.Value().RowView(i).(*mat.VecDense))
		copy(value[start:start+aCols], row)
	}
	for i := 0; i < bRows; i++ {
		start := i*(aCols+bCols) + aCols
		row := dupVec(op.b.Value().RowView(i).(*mat.VecDense))
		copy(value[start:start+bCols], row)
	}
	return mat.NewDense(bRows, aCols+bCols, value)
}

func (op *appendCol) op(grad *Tensor, a bool) {
	var target *Tensor
	var gradDense *mat.Dense
	var rows, cols int
	if a {
		target = op.a
		rows, cols = op.a.Dims()
		gradDense = grad.Value().Slice(0, rows, 0, cols).(*mat.Dense)
	} else {
		target = op.b
		_, aCols := op.a.Dims()
		rows, cols = op.b.Dims()
		gradDense = grad.Value().Slice(0, rows, aCols, aCols+cols).(*mat.Dense)
	}
	if cols == 0 {
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

func (op *appendCol) df(grad *Tensor) {
	if op.a.needGrad() {
		op.op(grad, true)
	}
	if op.b.needGrad() {
		op.op(grad, false)
	}
}

func (op *appendCol) ZeroGrad() {
	op.a.ZeroGrad()
	op.b.ZeroGrad()
}

func (op *appendCol) needGrad() bool {
	if op.a.needGrad() {
		return true
	}
	return op.b.needGrad()
}
