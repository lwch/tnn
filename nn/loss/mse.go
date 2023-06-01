package loss

import (
	"github.com/lwch/tnn/nn/tensor"
)

type MSE struct {
	*base
}

func NewMSE() *MSE {
	var loss MSE
	loss.base = new("mse")
	return &loss
}

func (*MSE) Loss(predict, targets *tensor.Tensor) *tensor.Tensor {
	rows, _ := predict.Dims()
	return predict.Sub(targets).Pow(2).Sum().Scale(1 / float32(rows))
}
