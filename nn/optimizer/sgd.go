package optimizer

import (
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
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
	ret := make([]*params.Params, len(grads))
	for i := 0; i < len(grads); i++ {
		ret[i] = params.New()
		grads[i].Range(func(name string, t *tensor.Tensor) {
			ret[i].Set(name, t.Grad().Scale(-sgd.lr))
		})
	}
	return ret
}
