package optimizer

import "gorgonia.org/gorgonia"

type Adam struct {
	*base
	slover gorgonia.Solver
}

func NewAdam(lr, l1reg, l2reg float64) Optimizer {
	var optimizer Adam
	optimizer.base = new("adam", lr, l1reg, l2reg)
	optimizer.slover = gorgonia.NewAdamSolver(
		gorgonia.WithLearnRate(lr),
		gorgonia.WithL1Reg(l1reg),
		gorgonia.WithL2Reg(l2reg),
	)
	return &optimizer
}

func (optimizer *Adam) Step(params gorgonia.Nodes) error {
	return optimizer.slover.Step(gorgonia.NodesToValueGrads(params))
}
