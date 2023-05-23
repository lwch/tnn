package optimizer

import (
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
)

type Optimizer interface {
	Update([]*params.Params)
	GetLr() float64
}

type computeFunc func(params []*params.Params) []*params.Params

type base struct {
	name        string
	lr          float64
	weightDecay float64
	computeFunc computeFunc
}

func new(name string, lr, weightDecay float64, compute computeFunc) *base {
	return &base{
		name:        name,
		lr:          lr,
		weightDecay: weightDecay,
		computeFunc: compute,
	}
}

func (opt *base) Update(params []*params.Params) {
	next := opt.computeFunc(params)
	if opt.weightDecay != 0 {
		for i := 0; i < len(next); i++ {
			next[i].Range(func(name string, t *tensor.Tensor) {
				scale := t.Scale(opt.lr * opt.weightDecay)
				next[i].Set(name, t.Sub(scale))
			})
		}
	}
	for i := 0; i < len(next); i++ {
		next[i].Range(func(name string, t *tensor.Tensor) {
			params[i].Get(name).AddValue(t.Value())
		})
	}
}

func (opt *base) GetLr() float64 {
	return opt.lr
}
