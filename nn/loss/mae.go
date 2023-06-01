package loss

import "github.com/lwch/tnn/nn/tensor"

type MAE struct {
	*base
}

func NewMAE() *MAE {
	var loss MAE
	loss.base = new("mae")
	return &loss
}

func (*MAE) Loss(predict, targets *tensor.Tensor) *tensor.Tensor {
	rows, _ := predict.Dims()
	return predict.Sub(targets).Sum().Scale(1 / float32(rows))
}
