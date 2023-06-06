package optimizer

import (
	"gorgonia.org/gorgonia"
)

type Adam struct {
	*base
	slover gorgonia.Solver
}

func NewAdam(options ...OptimizerOption) Optimizer {
	var optimizer Adam
	optimizer.base = newBase()
	for _, opt := range options {
		opt(optimizer.base)
	}
	var opts []gorgonia.SolverOpt
	opts = append(opts, gorgonia.WithLearnRate(optimizer.lr))
	if optimizer.l1reg > 0 {
		opts = append(opts, gorgonia.WithL1Reg(optimizer.l1reg))
	}
	if optimizer.l2reg > 0 {
		opts = append(opts, gorgonia.WithL2Reg(optimizer.l2reg))
	}
	optimizer.slover = gorgonia.NewAdamSolver(opts...)
	return &optimizer
}

func (optimizer *Adam) Step(params gorgonia.Nodes) error {
	return optimizer.slover.Step(gorgonia.NodesToValueGrads(params))
}
