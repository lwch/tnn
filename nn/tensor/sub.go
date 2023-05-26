package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type sub struct {
	a, b *Tensor
}

func dupVec(vec *mat.VecDense) []float64 {
	var data []float64
	data = append(data, vec.RawVector().Data...)
	return data
}

func (op *sub) Forward() *Tensor {
	aRows, aCols := op.a.Dims()
	bRows, bCols := op.b.Dims()
	if aRows == bRows && aCols != bCols { // 减去列向量
		if bCols != 1 {
			panic("bCols!=1")
		}
		ret := mat.NewDense(aRows, aCols, nil)
		for i := 0; i < aCols; i++ {
			var vec mat.VecDense
			vec.SubVec(op.a.Value().ColView(i), op.b.Value().ColView(0))
			ret.SetCol(i, dupVec(&vec))
		}
		return FromDense(ret)
	} else if aCols == bCols && aRows != bRows { // 减去行向量
		if bRows != 1 {
			panic("bRows!=1")
		}
		ret := mat.NewDense(aRows, aCols, nil)
		for i := 0; i < aRows; i++ {
			var vec mat.VecDense
			vec.SubVec(op.a.Value().RowView(i), op.b.Value().RowView(0))
			ret.SetRow(i, dupVec(&vec))
		}
		return FromDense(ret)
	} else {
		var value mat.Dense
		value.Sub(op.a.Value(), op.b.Value())
		return FromDense(&value)
	}
}

func (op *sub) Backward(grad *Tensor) {
	op.a.AddGrad(grad.Value())
	op.a.Backward(grad)
	gRows, gCols := grad.Dims()
	bRows, bCols := op.b.Dims()
	db := grad.Scale(-1)
	if gRows != bRows {
		v := mat.NewVecDense(gCols, nil)
		for i := 0; i < gRows; i++ {
			v.AddVec(v, grad.Value().RowView(i))
		}
		db = FromRowVector(v).Scale(-1)
	} else if gCols != bCols {
		v := mat.NewVecDense(gRows, nil)
		for i := 0; i < gCols; i++ {
			v.AddVec(v, grad.Value().ColView(i))
		}
		db = FromColVector(v).Scale(-1)
	}
	op.b.AddGrad(db.Value())
	op.b.Backward(db)
}

func (op *sub) Dims() (int, int) {
	return op.a.Dims()
}

func (op *sub) ZeroGrad() {
	op.a.ZeroGrad()
	op.b.ZeroGrad()
}
