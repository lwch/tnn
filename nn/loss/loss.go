package loss

import (
	"github.com/lwch/tnn/nn/tensor"
)

type Loss interface {
	Name() string
	Loss(predict, targets *tensor.Tensor) float64
	Grad(predict, targets *tensor.Tensor) *tensor.Tensor
}
