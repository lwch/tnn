package optimizer

import (
	"fmt"

	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/pb"
	"github.com/lwch/tnn/nn/vector"
	"gonum.org/v1/gonum/mat"
)

type Optimizer interface {
	Update(grads, params []*params.Params)
	Save() *pb.Optimizer
	Print()
}

type computeFunc func(grads []*params.Params) []*params.Params

type base struct {
	name        string
	lr          float64
	weightDecay float64
	compute     computeFunc
}

func new(name string, lr, weightDecay float64, compute computeFunc) *base {
	return &base{
		name:        name,
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
		panic("unsupported " + opt.GetName() + " optimizer")
	}
}

func (opt *base) Update(grads, params []*params.Params) {
	grads = opt.compute(grads)
	for i := 0; i < len(grads); i++ {
		for _, grad := range *grads[i] {
			var tmp mat.Dense
			tmp.Scale(opt.lr*opt.weightDecay, grad)
			grad.(vector.Suber).Sub(grad, &tmp)
		}
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

func (opt *base) Print() {
	fmt.Println("Optimizer:", opt.name)
	fmt.Println("  - Learning Rate:", opt.lr)
	fmt.Println("  - Weight Decay:", opt.weightDecay)
}
