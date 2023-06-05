package optimizer

import "gorgonia.org/gorgonia"

type Adam struct {
	slover gorgonia.Solver
}

func NewAdam(lr float64) Optimizer {
	return &Adam{
		slover: gorgonia.NewAdamSolver(
			gorgonia.WithLearnRate(lr),
		),
	}
}

func (optimizer *Adam) Step(params gorgonia.Nodes) error {
	return optimizer.slover.Step(gorgonia.NodesToValueGrads(params))
}
