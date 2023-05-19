package operator

import "github.com/lwch/tnn/nn/tensor"

type Operator interface {
	Forward() *tensor.Tensor
	Backward(grad *tensor.Tensor) []*tensor.Tensor
}
