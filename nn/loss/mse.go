package loss

import (
	"github.com/lwch/tnn/nn/tensor"
)

type MSE struct{}

func NewMSE() *MSE {
	return &MSE{}
}

func (*MSE) Name() string {
	return "mse"
}

func (*MSE) Loss(predict, targets *tensor.Tensor) *tensor.Tensor {
	rows, _ := predict.Dims()
	return predict.Sub(targets).Pow(2).Sum().Scale(1 / float64(rows))
}
