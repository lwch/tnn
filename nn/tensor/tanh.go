package tensor

import (
	"math"

	"github.com/lwch/gonum/mat32"
)

type tanh struct {
	a *Tensor
}

func tanhDense(x *mat32.Dense) *mat32.Dense {
	var value mat32.Dense
	value.Apply(func(i, j int, v float32) float32 {
		return float32(math.Tanh(float64(v)))
	}, x)
	return &value
}

func (op *tanh) f() *mat32.Dense {
	return tanhDense(op.a.Value())
}

func (op *tanh) df(grad *Tensor) {
	a := Ones(grad.Dims())
	b := powDense(tanhDense(grad.Value()), 2)
	var delta mat32.Dense
	delta.Sub(a.Value(), b)
	op.a.AddGrad(&delta)
	op.a.Backward(FromDense(&delta))
}

func (op *tanh) ZeroGrad() {
	op.a.ZeroGrad()
}

func (op *tanh) needGrad() bool {
	return op.a.needGrad()
}
