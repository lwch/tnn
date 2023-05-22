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
	sub := predict.Sub(targets)
	sub.SetName("mse.loss.diff")
	pow := sub.MulElem(sub)
	pow.SetName("mse.loss.pow")
	sum := pow.Sum()
	sum.SetName("mse.loss.sum")
	rows, _ := predict.Dims()
	return 0.5 * sum.Value().At(0, 0) / float64(rows)
}

func (*MSE) Grad(predict, targets *tensor.Tensor) *tensor.Tensor {
	sub := predict.Sub(targets)
	sub.SetName("mse.grad.diff")
	rows, _ := predict.Dims()
	scale := sub.Scale(1 / float64(rows))
	scale.SetName("mse.grad.scale")
	return scale
}
