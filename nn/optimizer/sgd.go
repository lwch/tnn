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

func (sgd *SGD) compute(grads *params.List) *params.List {
	ret := params.NewList()
	grads.Range(func(i int, t *tensor.Tensor) {
		ret.Add(t.Grad().Scale(-sgd.lr))
	})
	return ret
}
