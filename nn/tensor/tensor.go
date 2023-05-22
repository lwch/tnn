package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type Tensor struct {
	name string
	data *mat.Dense
	op   Operator
	grad *Tensor
}

func New(data []float64, rows, cols int) *Tensor {
	return &Tensor{data: mat.NewDense(rows, cols, data)}
}

func FromDense(dense *mat.Dense) *Tensor {
	return &Tensor{data: dense}
}

func FromVector(vector *mat.VecDense) *Tensor {
	var data []float64
	data = append(data, vector.RawVector().Data...)
	return &Tensor{data: mat.NewDense(1, vector.Len(), data)}
}

func (t *Tensor) SetName(name string) {
	t.name = name
}

func (t *Tensor) Value() *mat.Dense {
	if t.op != nil {
		return t.op.Forward().Value()
	}
	return t.data
}

func (t *Tensor) Clone() *Tensor {
	var data mat.Dense
	data.CloneFrom(t.data)
	return &Tensor{data: &data}
}

func (t *Tensor) Forward() *Tensor {
	if t.op == nil {
		return nil
	}
	return t.op.Forward()
}

func (t *Tensor) Backward(grad *Tensor) {
	if t.op == nil {
		return
	}
	t.op.Backward(grad)
}

func (t *Tensor) Grad() *Tensor {
	return t.grad
}

func (t *Tensor) Dims() (int, int) {
	if t.op != nil {
		return t.op.Dims()
	}
	return t.data.Dims()
}

func (t *Tensor) repeat(n int) *Tensor {
	rows, cols := t.Dims()
	if rows != 1 {
		panic("repeat only support vector")
	}
	value := mat.NewDense(n, cols, nil)
	for i := 0; i < n; i++ {
		value.RowView(i).(*mat.VecDense).CopyVec(t.data.RowView(0))
	}
	return FromDense(value)
}
