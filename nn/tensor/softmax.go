package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type softmax struct {
	a     *Tensor
	axis  int
	value mat.Dense
	diff  *mat.Dense
}

// exp(x) / sum(exp(max(x)))
func (op *softmax) f() *mat.Dense {
	max := op.a.MaxAxis(op.axis)
	exp := op.a.Sub(max).Exp()
	op.value.CloneFrom(exp.DivElem(exp.SumAxis(op.axis)).Value())
	var v mat.Vector
	switch op.axis {
	case 0:
		v = op.value.ColView(0)
	case 1:
		v = op.value.RowView(0)
	default:
		panic("invalid axis")
	}
	op.diff = mat.NewDense(v.Len(), v.Len(), nil)
	for i := 0; i < v.Len(); i++ {
		for j := 0; j < v.Len(); j++ {
			if i == j {
				op.diff.Set(i, j, v.AtVec(i)-v.AtVec(i)*v.AtVec(j))
			} else {
				op.diff.Set(i, j, -v.AtVec(i)*v.AtVec(j))
			}
		}
	}
	return &op.value
}

func (op *softmax) df(grad *Tensor) {
	var delta mat.Dense
	if op.axis == 0 {
		delta.Mul(op.diff, grad.Value())
	} else {
		delta.Mul(grad.Value(), op.diff)
	}
	op.a.AddGrad(&delta)
	op.a.Backward(FromDense(&delta))
}

func (op *softmax) ZeroGrad() {
	op.a.ZeroGrad()
}

func (op *softmax) needGrad() bool {
	return op.a.needGrad()
}
