package lr

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/optimizer"
)

type Exponential struct {
	*base
	n float64
}

func NewExponential(opt optimizer.Optimizer, n float64) *Exponential {
	var s Exponential
	s.base = new("exponential", opt.GetLr())
	s.n = n
	return &s
}

func (s *Exponential) Step(lr float64) float64 {
	s.currentLr = s.n * lr
	return s.currentLr
}

func (s *Exponential) Save() *pb.Scheduler {
	ret := s.base.Save()
	ret.Params = map[string]float64{
		"n": s.n,
	}
	return ret
}
