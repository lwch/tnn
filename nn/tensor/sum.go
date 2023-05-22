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
	op.a.grad = Numbers(rows, cols, grad.Value().At(0, 0))
	op.a.Backward(op.a.grad)
}

func (op *sum) Dims() (int, int) {
	return op.a.Dims()
}
