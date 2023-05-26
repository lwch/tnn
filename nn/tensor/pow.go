package tensor

import "gonum.org/v1/gonum/mat"

type pow struct {
	a *Tensor
	b float64
}

func (op *pow) f() *mat.Dense {
	value := powDense(op.a.Value(), op.b)
	return value
}

func (op *pow) df(grad *Tensor) {
	pow := powDense(op.a.Value(), op.b-1)
	pow.Scale(op.b, pow)
	op.a.AddGrad(pow)
	op.a.Backward(FromDense(pow))
}

func (op *pow) ZeroGrad() {
	op.a.ZeroGrad()
}
