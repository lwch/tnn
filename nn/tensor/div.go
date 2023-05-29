package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type divElem struct {
	a, b *Tensor
}

func (op *divElem) f() *mat.Dense {
	aRows, aCols := op.a.Dims()
	bRows, bCols := op.b.Dims()
	if aRows == bRows && aCols != bCols { // 减去列向量
		if bCols != 1 {
			panic("bCols!=1")
		}
		ret := mat.NewDense(aRows, aCols, nil)
		for i := 0; i < aCols; i++ {
			var vec mat.VecDense
			vec.DivElemVec(op.a.Value().ColView(i), op.b.Value().ColView(0))
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
			vec.DivElemVec(op.a.Value().RowView(i), op.b.Value().RowView(0))
			ret.SetRow(i, dupVec(&vec))
		}
		return ret
	} else {
		var value mat.Dense
		value.DivElem(op.a.Value(), op.b.Value())
		return &value
	}
}

func (op *divElem) df(grad *Tensor) {
	aRows, aCols := op.a.Dims()
	da := mat.NewDense(aRows, aCols, nil)
	gRows, gCols := grad.Dims()
	bRows, bCols := op.b.Dims()
	if gRows != bRows {
		b := op.b.Value().RowView(0)
		for i := 0; i < aRows; i++ {
			var v mat.VecDense
			v.DivElemVec(grad.Value().RowView(i), b)
			da.SetRow(i, dupVec(&v))
		}
	} else if gCols != bCols {
		b := op.b.Value().ColView(0)
		for i := 0; i < aCols; i++ {
			var v mat.VecDense
			v.DivElemVec(grad.Value().ColView(i), b)
			da.SetCol(i, dupVec(&v))
		}
	} else {
		da.DivElem(grad.Value(), op.b.Value())
	}
	op.a.AddGrad(da)
	op.a.Backward(FromDense(da))

	db := new(mat.Dense)
	db.Scale(-1, grad.Value())
	db.MulElem(db, op.a.Value())
	pow := powDense(op.b.Value(), 2)
	if gRows != bRows {
		sum := mat.NewVecDense(gCols, nil)
		for i := 0; i < gRows; i++ {
			sum.AddVec(sum, db.RowView(i))
		}
		sum.DivElemVec(sum, pow.RowView(0))
		db = mat.NewDense(bRows, gCols, dupVec(sum))
	} else if gCols != bCols {
		sum := mat.NewVecDense(gRows, nil)
		for i := 0; i < gCols; i++ {
			sum.AddVec(sum, db.ColView(i))
		}
		sum.DivElemVec(sum, pow.ColView(0))
		db = mat.NewDense(gRows, bCols, dupVec(sum))
	} else {
		db.DivElem(db, pow)
	}
	op.b.AddGrad(db)
	op.b.Backward(FromDense(db))
}

func (op *divElem) ZeroGrad() {
	op.a.ZeroGrad()
	op.b.ZeroGrad()
}
