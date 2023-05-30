package math

import (
	"github.com/lwch/tnn/nn/tensor"
)

// Sigmoid 1 / (1 + exp(-x))
func Sigmoid(x *tensor.Tensor) *tensor.Tensor {
	one := tensor.Ones(x.Dims())
	return x.Scale(-1).Exp().Add(one).Inv()
}

// Softmax exp(x) / sum(exp(max(x)))
func Softmax(x *tensor.Tensor, axis int) *tensor.Tensor {
	max := x.MaxAxis(axis)
	exp := x.Sub(max).Exp()
	expValue := tensor.FromDense(exp.Value()) // 截断反向传播
	return expValue.DivElem(expValue.SumAxis(axis))
}

// LogSoftmax x - max(x) - log(sum(exp(x - max(x))))
func LogSoftmax(x *tensor.Tensor, axis int) *tensor.Tensor {
	max := x.MaxAxis(axis)
	exp := x.Sub(max).Exp()
	sum := exp.SumAxis(axis)
	return x.Sub(max).Sub(sum.Log())
}
