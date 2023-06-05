package optimizer

import (
	"gorgonia.org/gorgonia"
)

type Optimizer interface {
	Step(params gorgonia.Nodes) error
}

type OptimizerOption func(*base)

type base struct {
	lr           float64
	l1reg, l2reg float64
}

func newBase() *base {
	return &base{lr: 0.001}
}

func WithLearnRate(lr float64) OptimizerOption {
	return func(optimizer *base) {
		optimizer.lr = lr
	}
}

func WithL1Reg(l1reg float64) OptimizerOption {
	return func(optimizer *base) {
		optimizer.l1reg = l1reg
	}
}

func WithL2Reg(l2reg float64) OptimizerOption {
	return func(optimizer *base) {
		optimizer.l2reg = l2reg
	}
}
