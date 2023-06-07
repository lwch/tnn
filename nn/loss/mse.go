package loss

import (
	"github.com/sugarme/gotch/ts"
)

type MSE struct {
}

func NewMSE() Loss {
	return &MSE{}
}

func (mse *MSE) Loss(y, pred *ts.Tensor) *ts.Tensor {
	return y.MustMseLoss(pred, ts.ReductionMean, true)
}
