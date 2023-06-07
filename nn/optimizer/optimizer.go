package optimizer

import (
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

type Optimizer interface {
	Step(vs *nn.VarStore, loss *ts.Tensor) error
}

type OptimizerOption func(*base)

type base struct {
	lr          float64
	weightDeacy float64
}

func newBase() *base {
	return &base{lr: 0.001}
}

func WithLearnRate(lr float64) OptimizerOption {
	return func(optimizer *base) {
		optimizer.lr = lr
	}
}

func WithWeightDeacy(wd float64) OptimizerOption {
	return func(optimizer *base) {
		optimizer.weightDeacy = wd
	}
}
