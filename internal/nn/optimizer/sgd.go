package optimizer

import "gonum.org/v1/gonum/mat"

type SGD struct {
	lr, weightDecay float64
}

func NewSGD(lr, weightDecay float64) *SGD {
	return &SGD{lr, weightDecay}
}

func (sgd *SGD) Update(weights, delta *mat.Dense) {
	delta.Scale(-sgd.lr, delta)
	weights.Add(weights, delta)
}
