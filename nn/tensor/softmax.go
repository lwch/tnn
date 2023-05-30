package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type softmax struct {
	a     *Tensor
	axis  int
	value mat.Dense
}

// exp(x) / sum(exp(max(x)))
func (op *softmax) f() *mat.Dense {
	max := op.a.MaxAxis(op.axis)
	exp := op.a.Sub(max).Exp()
	op.value.CloneFrom(exp.DivElem(exp.SumAxis(op.axis)).Value())
	return &op.value
}

func (op *softmax) df(grad *Tensor) {
	var v mat.Vector
	switch op.axis {
	case 0:
		v = op.value.ColView(0)
	case 1:
		v = op.value.RowView(0)
	default:
		panic("invalid axis")
	}
	diff := mat.NewDense(v.Len(), v.Len(), nil)
	for i := 0; i < v.Len(); i++ {
		for j := 0; j < v.Len(); j++ {
			if i == j {
				diff.Set(i, j, v.AtVec(i)-v.AtVec(i)*v.AtVec(j))
			} else {
				diff.Set(i, j, -v.AtVec(i)*v.AtVec(j))
			}
		}
	}
	var delta mat.Dense
	// fmt.Println(grad.Dims())
	// fmt.Println(diff.Dims())
	if op.axis == 0 {
		delta.Mul(diff, grad.Value())
	} else {
		delta.Mul(grad.Value(), diff)
	}
	op.a.AddGrad(&delta)
	op.a.Backward(FromDense(&delta))
}

func (op *softmax) ZeroGrad() {
	op.a.ZeroGrad()
}
