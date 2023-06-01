package tensor

import "github.com/lwch/gonum/mat32"

type scale struct {
	a float32
	b *Tensor
}

func (op *scale) f() *mat32.Dense {
	var value mat32.Dense
	value.Scale(op.a, op.b.Value())
	return &value
}

func (op *scale) df(grad *Tensor) {
	if !op.b.needGrad() {
		return
	}
	var delta mat32.Dense
	delta.Scale(op.a, grad.Value())
	op.b.AddGrad(&delta)
	op.b.Backward(FromDense(&delta))
}

func (op *scale) ZeroGrad() {
	op.b.ZeroGrad()
}

func (op *scale) needGrad() bool {
	return op.b.needGrad()
}
