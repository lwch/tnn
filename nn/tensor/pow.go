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
	delta := grad.MulElem(FromDense(pow))
	op.a.AddGrad(delta.Value())
	op.a.Backward(delta)
}

func (op *pow) ZeroGrad() {
	op.a.ZeroGrad()
}
