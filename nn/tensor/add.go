package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type add struct {
	a, b *Tensor
}

func (op *add) f() *mat.Dense {
	var value mat.Dense
	value.Add(op.a.Value(), op.b.Value())
	return &value
}

func (op *add) df(grad *Tensor) {
	op.a.AddGrad(grad.Value())
	op.b.AddGrad(grad.Value())
	op.a.Backward(grad)
	op.b.Backward(grad)
}

func (op *add) ZeroGrad() {
	op.a.ZeroGrad()
	op.b.ZeroGrad()
}

type addVector struct {
	a, b *Tensor
}

func (op *addVector) f() *mat.Dense {
	av := op.a.Value()
	rows, cols := av.Dims()
	b0 := op.b.Value().RowView(0)
	value := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		value.RowView(i).(*mat.VecDense).AddVec(av.RowView(i), b0)
	}
	return value
}

func (op *addVector) df(grad *Tensor) {
	gv := grad.Value()
	rows, cols := gv.Dims()
	delta := mat.NewVecDense(cols, nil)
	for i := 0; i < rows; i++ {
		delta.AddVec(delta, gv.RowView(i))
	}
	delta.ScaleVec(1/float64(rows), delta)
	op.a.AddGrad(grad.Value())
	op.b.AddGrad(vec2Dense(delta))
	op.a.Backward(grad)
	op.b.Backward(FromRowVector(delta))
}

func (op *addVector) ZeroGrad() {
	op.a.ZeroGrad()
	op.b.ZeroGrad()
}
