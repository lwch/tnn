package optimizer

import (
	"github.com/lwch/runtime"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

type Adam struct {
	*base
	cfg *nn.AdamConfig
}

func NewAdam(options ...OptimizerOption) Optimizer {
	var optimizer Adam
	optimizer.base = newBase()
	for _, opt := range options {
		opt(optimizer.base)
	}
	optimizer.cfg = nn.DefaultAdamConfig()
	optimizer.cfg.Wd = optimizer.weightDeacy
	return &optimizer
}

func (optimizer *Adam) Step(vs *nn.VarStore, loss *ts.Tensor) error {
	opt, err := optimizer.cfg.Build(vs, optimizer.lr)
	runtime.Assert(err)
	return opt.BackwardStep(loss)
}
