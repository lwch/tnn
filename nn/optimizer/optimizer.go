package optimizer

import (
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
)

type Optimizer interface {
	Update(*params.List)
	GetLr() float64
}

type computeFunc func(list *params.List) *params.List

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

func (opt *base) Update(params *params.List) {
	next := opt.computeFunc(params)
	if opt.weightDecay != 0 {
		next.Range(func(i int, t *tensor.Tensor) {
			scale := t.Scale(opt.lr * opt.weightDecay)
			next.Set(i, t.Sub(scale))
		})
	}
	next.Range(func(i int, t *tensor.Tensor) {
		params.Get(i).AddValue(t.Value())
	})
}

func (opt *base) GetLr() float64 {
	return opt.lr
}
