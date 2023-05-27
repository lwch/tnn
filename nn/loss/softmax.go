package loss

import (
	"github.com/lwch/tnn/nn/tensor"
)

type Softmax struct {
	*base
}

func NewSoftmax() *Softmax {
	var loss Softmax
	loss.base = new("softmax")
	return &loss
}

func (*Softmax) Loss(predict, targets *tensor.Tensor) *tensor.Tensor {
	rows, _ := predict.Dims()
	max := predict.MaxAxis(1)
	exps := predict.Sub(max).Exp()
	sum := exps.SumAxis(1)
	logSoftmax := predict.Sub(max).Sub(sum.Log())
	return logSoftmax.MulElem(targets).Scale(-1).Sum().Scale(1 / float64(rows))
}
