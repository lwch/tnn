package optimizer

import (
	"tnn/internal/nn/params"
	"tnn/internal/nn/pb"
)

type Optimizer interface {
	Update(grads, params []*params.Params)
	Save() *pb.Optimizer
}

type computeFunc func(grads []*params.Params) []*params.Params

type base struct {
	lr          float64
	weightDecay float64
	compute     computeFunc
}

func new(lr, weightDecay float64, compute computeFunc) *base {
	return &base{
		lr:          lr,
		weightDecay: weightDecay,
		compute:     compute,
	}
}

func Load(opt *pb.Optimizer) Optimizer {
	switch opt.Name {
	case "sgd":
		return NewSGD(opt.GetLr(), opt.GetWeightDecay())
	case "adam":
		ps := opt.GetParams()
		beta1 := ps["beta1"]
		beta2 := ps["beta2"]
		epsilon := ps["epsilon"]
		return NewAdam(opt.GetLr(), opt.GetWeightDecay(), beta1, beta2, epsilon)
	default:
		return nil
	}
}

func (opt *base) Update(grads, params []*params.Params) {
	grads = opt.compute(grads)
	for i := 0; i < len(grads); i++ {
		grads[i].Apply(func(i, j int, v float64) float64 {
			return v - opt.lr*opt.weightDecay*v
		})
	}
	for i := 0; i < len(params); i++ {
		if params[i] == nil {
			continue
		}
		params[i].Add(grads[i])
	}
}

func (opt *base) Save() *pb.Optimizer {
	return &pb.Optimizer{
		Lr:          opt.lr,
		WeightDecay: opt.weightDecay,
	}
}
