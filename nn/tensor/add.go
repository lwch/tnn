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
	op.a.grad = grad.Clone()
	op.b.grad = grad.Clone()
	op.a.Backward(op.a.grad)
	op.b.Backward(op.b.grad)
}

func (op *add) Dims() (int, int) {
	return op.a.Dims()
}

type addVector struct {
	a, b *Tensor
}

func (op *addVector) Forward() *Tensor {
	av := op.a.Value()
	rows, cols := av.Dims()
	value := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		value.RowView(i).(*mat.VecDense).AddVec(av.RowView(i), op.b.Value().RowView(0))
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
	op.a.grad = FromVector(delta)
	op.b.grad = FromVector(delta)
	op.a.Backward(op.a.grad.repeat(rows))
	op.b.Backward(op.b.grad.repeat(rows))
}

func (op *addVector) Dims() (int, int) {
	return op.a.Dims()
}
