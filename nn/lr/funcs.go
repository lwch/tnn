package lr

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/optimizer"
)

type Funcs struct {
	*base
	funcs []Func
}

type Func func(float64) float64

func NewFuncs(opt optimizer.Optimizer, fns ...Func) *Funcs {
	var s Funcs
	s.base = new("funcs", opt.GetLr())
	s.funcs = fns
	return &s
}

func (s *Funcs) Step(lr float64) float64 {
	for _, fn := range s.funcs {
		lr = fn(lr)
	}
	s.currentLr = lr
	return lr
}

func (s *Funcs) Save() *pb.Scheduler {
	return nil
}
