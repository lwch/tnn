package tensor

import "gonum.org/v1/gonum/mat"

type sum struct {
	a *Tensor
}

func (op *sum) Forward() *Tensor {
	n := mat.Sum(op.a.Value())
	return New([]float64{n}, 1, 1)
}

func (op *sum) Backward(grad *Tensor) {
	rows, cols := op.a.Value().Dims()
	delta := Numbers(rows, cols, grad.Value().At(0, 0))
	if op.a.grad == nil {
		op.a.grad = Zeros(rows, cols)
	}
	op.a.grad.AddValue(delta.Value())
	op.a.Backward(delta)
}

func (op *sum) Dims() (int, int) {
	return op.a.Dims()
}

func (op *sum) ZeroGrad() {
	op.a.ZeroGrad()
}
