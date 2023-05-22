package math

import "github.com/lwch/tnn/nn/tensor"

// Sigmoid 1 / (1 + exp(-x))
func Sigmoid(x *tensor.Tensor) *tensor.Tensor {
	one1 := tensor.Ones(x.Dims())
	neg := x.Scale(-1)
	neg.SetName("sigmoid.neg")
	exp := neg.Exp()
	exp.SetName("sigmoid.exp")
	add := exp.Add(one1)
	add.SetName("sigmoid.add")
	inv := add.Inv()
	inv.SetName("sigmoid.inv")
	return inv
}
