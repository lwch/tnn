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
	rows, cols := weights.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			weights.Set(i, j, weights.At(i, j)-sgd.lr*sgd.weightDecay*weights.At(i, j))
		}
	}
}
