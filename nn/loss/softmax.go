package loss

import (
	"github.com/lwch/tnn/internal/math"
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
	max := math.Max(predict, 1)
	exps := predict.Sub(tensor.FromColVector(max)).Exp()
	sum := exps.SumAxis(1)
	logSoftmax := predict.Sub(tensor.FromColVector(max)).Sub(sum.Log())
	return logSoftmax.MulElem(targets).Sum().Scale(1 / float64(rows))
}
