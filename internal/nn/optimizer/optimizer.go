package optimizer

import (
	"tnn/internal/nn/params"
)

type Optimizer interface {
	Update(grads, params []*params.Params)
}

type computeFunc func(grads []*params.Params) []*params.Params

type base struct {
	lr          float64
	weightDecay float64
	compute     computeFunc
}

func new(lr, weightDecay float64, compute computeFunc) *base {
	return &base{
		lr:          lr,
		weightDecay: weightDecay,
		compute:     compute,
	}
}

func (opt *base) Update(grads, params []*params.Params) {
	grads = opt.compute(grads)
	for i := 0; i < len(grads); i++ {
		grads[i].Apply(func(i, j int, v float64) float64 {
			return v - opt.lr*opt.weightDecay*v
		})
	}
	for i := 0; i < len(params); i++ {
		if params[i] == nil {
			continue
		}
		params[i].Add(grads[i])
	}
}
