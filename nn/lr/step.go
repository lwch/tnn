package lr

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/optimizer"
)

type Step struct {
	*base
	currentStep int
	stepSize    int
	gamma       float64
}

func NewStep(opt optimizer.Optimizer, stepSize int, gamma float64) *Step {
	var s Step
	s.base = new("step", opt.GetLr())
	s.stepSize = stepSize
	s.gamma = gamma
	return &s
}

func (s *Step) Step(lr float64) float64 {
	s.currentStep++
	if s.currentStep%s.stepSize == 0 {
		lr *= s.gamma
	}
	s.currentLr = lr
	return lr
}

func (s *Step) Save() *pb.Scheduler {
	ret := s.base.Save()
	ret.Params = map[string]float64{
		"stepSize": float64(s.stepSize),
		"gamma":    s.gamma,
		"step":     float64(s.currentStep),
	}
	return ret
}

func (s *Step) setStep(n int) {
	s.currentStep = n
}
