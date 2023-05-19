package tensor

import "gonum.org/v1/gonum/mat"

type Tensor struct {
	data *mat.Dense
	op   Operator
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

func (t *Tensor) Backward(grad *Tensor) []*Tensor {
	if t.op == nil {
		return nil
	}
	return t.op.Backward(grad)
}

func (t *Tensor) Dims() (int, int) {
	if t.op != nil {
		return t.op.Dims()
	}
	return t.data.Dims()
}

func (t *Tensor) isLeaf() bool {
	return t.op == nil
}
