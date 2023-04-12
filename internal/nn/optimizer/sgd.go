package optimizer

import (
	"tnn/internal/nn/params"
	"tnn/internal/nn/pb"
)

type SGD struct {
	*base
}

func NewSGD(lr, weightDecay float64) *SGD {
	var sgd SGD
	sgd.base = new(lr, weightDecay, sgd.compute)
	return &sgd
}

func (sgd *SGD) compute(grads []*params.Params) []*params.Params {
	for i := 0; i < len(grads); i++ {
		params := grads[i]
		for _, grad := range *params {
			grad.Scale(-sgd.lr, grad)
		}
	}
	return grads
}

func (sgd *SGD) Save() *pb.Optimizer {
	return sgd.base.Save()
}
