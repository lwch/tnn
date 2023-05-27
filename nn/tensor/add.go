package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type add struct {
	a, b *Tensor
}

func (op *add) f() *mat.Dense {
	aRows, aCols := op.a.Dims()
	bRows, bCols := op.b.Dims()
	if aRows == bRows && aCols != bCols { // 减去列向量
		if bCols != 1 {
			panic("bCols!=1")
		}
		ret := mat.NewDense(aRows, aCols, nil)
		for i := 0; i < aCols; i++ {
			var vec mat.VecDense
			vec.AddVec(op.a.Value().ColView(i), op.b.Value().ColView(0))
			ret.SetCol(i, dupVec(&vec))
		}
		return ret
	} else if aCols == bCols && aRows != bRows { // 减去行向量
		if bRows != 1 {
			panic("bRows!=1")
		}
		ret := mat.NewDense(aRows, aCols, nil)
		for i := 0; i < aRows; i++ {
			var vec mat.VecDense
			vec.AddVec(op.a.Value().RowView(i), op.b.Value().RowView(0))
			ret.SetRow(i, dupVec(&vec))
		}
		return ret
	} else {
		var value mat.Dense
		value.Add(op.a.Value(), op.b.Value())
		return &value
	}
}

func (op *add) df(grad *Tensor) {
	op.a.AddGrad(grad.Value())
	op.a.Backward(grad)
	gRows, gCols := grad.Dims()
	bRows, bCols := op.b.Dims()
	db := grad.Value()
	if gRows != bRows {
		sum := mat.NewVecDense(gCols, nil)
		for i := 0; i < gRows; i++ {
			sum.AddVec(sum, grad.Value().RowView(i))
		}
		db = mat.NewDense(bRows, bCols, dupVec(sum))
	} else if gCols != bCols {
		sum := mat.NewVecDense(gRows, nil)
		for i := 0; i < gCols; i++ {
			sum.AddVec(sum, grad.Value().ColView(i))
		}
		db = mat.NewDense(bRows, bCols, dupVec(sum))
	}
	op.b.AddGrad(db)
	op.b.Backward(FromDense(db))
}

func (op *add) ZeroGrad() {
	op.a.ZeroGrad()
	op.b.ZeroGrad()
}
