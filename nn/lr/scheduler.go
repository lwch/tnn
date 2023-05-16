package lr

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/optimizer"
)

type Scheduler interface {
	Step(float64) float64
	Save() *pb.Scheduler
}

type base struct {
	name      string
	initLr    float64
	currentLr float64
}

func new(name string, initLr float64) *base {
	return &base{
		name:   name,
		initLr: initLr,
	}
}

func Load(s *pb.Scheduler, opt optimizer.Optimizer) Scheduler {
	opt.SetLr(s.GetCurrent())
	var ret Scheduler
	switch s.Name {
	case "funcs":
		return nil
	case "step":
		ps := s.GetParams()
		stepSize := ps["stepSize"]
		gamma := ps["gamma"]
		ret = NewStep(opt, int(stepSize), gamma)
		step := ps["step"]
		ret.(*Step).setStep(int(step))
	case "exponential":
		ps := s.GetParams()
		n := ps["n"]
		ret = NewExponential(opt, n)
	default:
		panic("unsupported " + s.GetName() + " scheduler")
	}
	type setInit interface {
		setInit(float64)
	}
	ret.(setInit).setInit(s.GetInit())
	return ret
}

func (opt *base) Save() *pb.Scheduler {
	return &pb.Scheduler{
		Name:    opt.name,
		Init:    opt.initLr,
		Current: opt.currentLr,
	}
}

func (opt *base) Step(lr float64) float64 {
	panic("not implemented")
}

func (opt *base) setInit(lr float64) {
	opt.initLr = lr
}
