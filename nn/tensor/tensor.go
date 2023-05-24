package tensor

import (
	"sync"

	"gonum.org/v1/gonum/mat"
)

type Tensor struct {
	name  string
	data  *mat.Dense
	op    Operator
	gradM sync.Mutex
	grad  *Tensor
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

func (t *Tensor) Name() string {
	return t.name
}

func (t *Tensor) Value() *mat.Dense {
	if t.op != nil {
		return t.op.Forward().Value()
	}
	return t.data
}

func (t *Tensor) Clone() *Tensor {
	if t.op != nil {
		panic("clone only support value tensor")
	}
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

func (t *Tensor) ZeroGrad() {
	t.gradM.Lock()
	if t.grad != nil {
		t.grad.Zero()
	}
	t.gradM.Unlock()
	if t.op != nil {
		t.op.ZeroGrad()
	}
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

func (t *Tensor) AddValue(v *mat.Dense) {
	t.data.Add(t.data, v)
}

func (t *Tensor) AddGrad(v *mat.Dense) {
	t.gradM.Lock()
	defer t.gradM.Unlock()
	if t.grad == nil {
		t.grad = Zeros(v.Dims())
	}
	t.grad.AddValue(v)
}

func (t *Tensor) Zero() {
	t.data.Zero()
}

func (t *Tensor) Set(i, j int, v float64) {
	t.data.Set(i, j, v)
}
