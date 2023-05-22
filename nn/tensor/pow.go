package tensor

type pow struct {
	a *Tensor
	b float64
}

func (op *pow) Forward() *Tensor {
	value := powDense(op.a.Value(), op.b)
	return FromDense(value)
}

func (op *pow) Backward(grad *Tensor) {
	pow := powDense(op.a.Value(), op.b-1)
	pow.Scale(op.b, pow)
	op.a.grad = FromDense(pow)
	op.a.Backward(op.a.grad)
}

func (op *pow) Dims() (int, int) {
	return op.a.Dims()
}
