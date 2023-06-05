package optimizer

import (
	"gorgonia.org/gorgonia"
)

type Optimizer interface {
	Step(params gorgonia.Nodes) error
}

type base struct {
	name         string
	lr           float64
	l1reg, l2reg float64
}

func new(name string, lr, l1reg, l2reg float64) *base {
	return &base{
		name:  name,
		lr:    lr,
		l1reg: l1reg,
		l2reg: l2reg,
	}
}
