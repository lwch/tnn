package optimizer

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
)

type Optimizer interface {
	Update([]*params.Params)
	GetLr() float32
	Save() *pb.Optimizer
}

type computeFunc func(params []*params.Params) []*params.Params

type base struct {
	name        string
	lr          float32
	weightDecay float32
	computeFunc computeFunc
}

func new(name string, lr, weightDecay float32, compute computeFunc) *base {
	return &base{
		name:        name,
		lr:          lr,
		weightDecay: weightDecay,
		computeFunc: compute,
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
	// case "adagrad":
	// 	ps := opt.GetParams()
	// 	epsilon := ps["epsilon"]
	// 	return NewAdagrad(opt.GetLr(), opt.GetWeightDecay(), epsilon)
	default:
		panic("unsupported " + opt.GetName() + " optimizer")
	}
}

func (opt *base) Update(params []*params.Params) {
	next := opt.computeFunc(params)
	if opt.weightDecay != 0 {
		for i := 0; i < len(next); i++ {
			next[i].Range(func(name string, t *tensor.Tensor) {
				scale := t.Scale(opt.lr * opt.weightDecay)
				next[i].Set(name, t.Sub(scale))
			})
		}
	}
	for i := 0; i < len(next); i++ {
		next[i].Range(func(name string, t *tensor.Tensor) {
			params[i].Get(name).AddValue(t.Value())
		})
	}
}

func (opt *base) GetLr() float32 {
	return opt.lr
}

func (opt *base) Save() *pb.Optimizer {
	return &pb.Optimizer{
		Name:        opt.name,
		Lr:          opt.lr,
		WeightDecay: opt.weightDecay,
	}
}
