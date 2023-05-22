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

func (opt *base) Update(grads *params.List) {
	next := opt.computeFunc(grads)
	if opt.weightDecay != 0 {
		next.Range(func(i int, grad *tensor.Tensor) {
			scale := grad.Scale(opt.lr * opt.weightDecay)
			next.Set(i, grad.Sub(scale))
		})
	}
	grads.Range(func(i int, grad *tensor.Tensor) {
		grad.AddValue(next.Get(i).Value())
	})
}

func (opt *base) GetLr() float64 {
	return opt.lr
}
