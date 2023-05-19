package operator

import (
	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
)

type add struct {
	a, b *tensor.Tensor
}

func Add(a, b *tensor.Tensor) Operator {
	if !a.IsSameShape(b) {
		panic("invalid shape")
	}
	return &add{a, b}
}

func (o *add) Forward() *tensor.Tensor {
	var value mat.VecDense
	value.AddVec(o.a.Value(), o.b.Value())
	return tensor.FromVec(&value, o.a.Shapes()...)
}

func (o *add) Backward(grad *tensor.Tensor) []*tensor.Tensor {
	if grad == nil {
		grad = tensor.Ones(o.a.Shapes()...)
	}
	return []*tensor.Tensor{
		grad.Clone(),
		grad.Clone(),
	}
}
