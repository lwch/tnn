package optimizer

import "gonum.org/v1/gonum/mat"

type SGD struct {
	*base
}

func NewSGD(lr, weightDecay float64) *SGD {
	var sgd SGD
	sgd.base = new(lr, weightDecay, sgd.compute)
	return &sgd
}

func (sgd *SGD) compute(grads []*mat.Dense) []*mat.Dense {
	for i := 0; i < len(grads); i++ {
		grads[i].Scale(-sgd.lr, grads[i])
	}
	return grads
}

// func (sgd *SGD) Update(weights, delta *mat.Dense) {
// 	rows, cols := weights.Dims()
// 	for i := 0; i < rows; i++ {
// 		for j := 0; j < cols; j++ {
// 			d := sgd.compute(delta.At(i, j))
// 			d -= sgd.weightDecay.compute(d, sgd.lr)
// 			delta.Set(i, j, d)
// 			weights.Set(i, j, weights.At(i, j)+d)
// 		}
// 	}
// }

// func (sgd *SGD) compute(delta float64) float64 {
// 	return -sgd.lr * delta
// }
