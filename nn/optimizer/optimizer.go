package optimizer

import (
	"github.com/lwch/tnn/internal/pb"
	"gorgonia.org/gorgonia"
)

type Optimizer interface {
	Step(params gorgonia.Nodes) error
	Save() *pb.Optimizer
}

func Load(opt *pb.Optimizer) Optimizer {
	switch opt.Name {
	// case "sgd":
	// 	return NewSGD(opt.GetLr(), opt.GetWeightDecay())
	case "adam":
		return NewAdam(opt.GetLr(), opt.GetL1Reg(), opt.GetL2Reg())
	// case "adagrad":
	// 	ps := opt.GetParams()
	// 	epsilon := ps["epsilon"]
	// 	return NewAdagrad(opt.GetLr(), opt.GetWeightDecay(), epsilon)
	default:
		panic("unsupported " + opt.GetName() + " optimizer")
	}
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

func (opt *base) Save() *pb.Optimizer {
	return &pb.Optimizer{
		Name:  opt.name,
		Lr:    opt.lr,
		L1Reg: opt.l1reg,
		L2Reg: opt.l2reg,
	}
}
