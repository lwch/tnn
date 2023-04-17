package optimizer

import (
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/pb"
	"github.com/lwch/tnn/nn/vector"
)

type SGD struct {
	*base
}

func NewSGD(lr, weightDecay float64) *SGD {
	var sgd SGD
	sgd.base = new("sgd", lr, weightDecay, sgd.compute)
	return &sgd
}

func (sgd *SGD) compute(grads []*params.Params) []*params.Params {
	for i := 0; i < len(grads); i++ {
		params := grads[i]
		for _, grad := range *params {
			grad.(vector.Scaler).Scale(-sgd.lr, grad)
		}
	}
	return grads
}

func (sgd *SGD) Save() *pb.Optimizer {
	return sgd.base.Save()
}
