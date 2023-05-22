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

func (*MSE) Loss(predict, targets *tensor.Tensor) float64 {
	sum := predict.Sub(targets).Pow(2).Sum()
	rows, _ := predict.Dims()
	return 0.5 * sum.Value().At(0, 0) / float64(rows)
}

func (*MSE) Grad(predict, targets *tensor.Tensor) *tensor.Tensor {
	rows, _ := predict.Dims()
	return predict.Sub(targets).Scale(1 / float64(rows))
}
