package tensor

import (
	"github.com/lwch/gonum/mat32"
)

type sub struct {
	a, b *Tensor
}

func dupVec(vec *mat32.VecDense) []float32 {
	var data []float32
	data = append(data, vec.RawVector().Data...)
	return data
}

func (op *sub) f() *mat32.Dense {
	aRows, aCols := op.a.Dims()
	bRows, bCols := op.b.Dims()
	if aRows == bRows && aCols != bCols { // 减去列向量
		if bCols != 1 {
			panic("bCols!=1")
		}
		ret := mat32.NewDense(aRows, aCols, nil)
		for i := 0; i < aCols; i++ {
			var vec mat32.VecDense
			vec.SubVec(op.a.Value().ColView(i), op.b.Value().ColView(0))
			ret.SetCol(i, dupVec(&vec))
		}
		return ret
	} else if aCols == bCols && aRows != bRows { // 减去行向量
		if bRows != 1 {
			panic("bRows!=1")
		}
		ret := mat32.NewDense(aRows, aCols, nil)
		for i := 0; i < aRows; i++ {
			var vec mat32.VecDense
			vec.SubVec(op.a.Value().RowView(i), op.b.Value().RowView(0))
			ret.SetRow(i, dupVec(&vec))
		}
		return ret
	} else {
		var value mat32.Dense
		value.Sub(op.a.Value(), op.b.Value())
		return &value
	}
}

func (op *sub) df(grad *Tensor) {
	if op.a.needGrad() {
		op.a.AddGrad(grad.Value())
		op.a.Backward(grad)
	}
	if op.b.needGrad() {
		gRows, gCols := grad.Dims()
		bRows, bCols := op.b.Dims()
		// TODO: 优化
		db := grad.Scale(-1)
		if gRows != bRows {
			v := mat32.NewVecDense(gCols, nil)
			for i := 0; i < gRows; i++ {
				v.AddVec(v, grad.Value().RowView(i))
			}
			db = FromRowVector(v).Scale(-1)
		} else if gCols != bCols {
			v := mat32.NewVecDense(gRows, nil)
			for i := 0; i < gCols; i++ {
				v.AddVec(v, grad.Value().ColView(i))
			}
			db = FromColVector(v).Scale(-1)
		}
		op.b.AddGrad(db.Value())
		op.b.Backward(db)
	}
}

func (op *sub) ZeroGrad() {
	op.a.ZeroGrad()
	op.b.ZeroGrad()
}

func (op *sub) needGrad() bool {
	if op.a.needGrad() {
		return true
	}
	return op.b.needGrad()
}
