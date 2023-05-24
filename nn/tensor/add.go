package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type add struct {
	a, b *Tensor
}

func (op *add) Forward() *Tensor {
	var value mat.Dense
	value.Add(op.a.Value(), op.b.Value())
	return FromDense(&value)
}

func (op *add) Backward(grad *Tensor) {
	op.a.AddGrad(grad.Value())
	op.b.AddGrad(grad.Value())
	op.a.Backward(grad.Clone())
	op.b.Backward(grad.Clone())
}

func (op *add) Dims() (int, int) {
	return op.a.Dims()
}

func (op *add) ZeroGrad() {
	op.a.ZeroGrad()
	op.b.ZeroGrad()
}

type addVector struct {
	a, b *Tensor
}

func (op *addVector) Forward() *Tensor {
	av := op.a.Value()
	rows, cols := av.Dims()
	b0 := op.b.Value().RowView(0)
	value := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		value.RowView(i).(*mat.VecDense).AddVec(av.RowView(i), b0)
	}
	return FromDense(value)
}

func (op *addVector) Backward(grad *Tensor) {
	gv := grad.Value()
	rows, cols := gv.Dims()
	delta := mat.NewVecDense(cols, nil)
	for i := 0; i < rows; i++ {
		delta.AddVec(delta, gv.RowView(i))
	}
	delta.ScaleVec(1/float64(rows), delta)
	op.a.AddGrad(grad.Value())
	op.b.AddGrad(vec2Dense(delta))
	op.a.Backward(grad.Clone())
	op.b.Backward(FromVector(delta))
}

func (op *addVector) Dims() (int, int) {
	return op.a.Dims()
}

func (op *addVector) ZeroGrad() {
	op.a.ZeroGrad()
	op.b.ZeroGrad()
}
