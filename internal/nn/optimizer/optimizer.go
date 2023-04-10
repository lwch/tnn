package optimizer

import (
	"tnn/internal/nn/layer"

	"gonum.org/v1/gonum/mat"
)

type Optimizer interface {
	Update(params []*layer.Params)
}

type computeFunc func(grads []*mat.Dense) []*mat.Dense

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

func (opt *base) Update(params []*layer.Params) {
	// TODO
	grads = opt.compute(grads)
	for i := 0; i < len(grads); i++ {
		// var grad mat.Dense
		// grad.Apply(func(i, j int, v float64) float64 {
		// }, grads[i])
	}
	for i := 0; i < len(params); i++ {
		params[i].Add(grads[i])
	}
}
