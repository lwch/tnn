package optimizer

import "gonum.org/v1/gonum/mat"

type Optimizer interface {
	Update(weights, delta *mat.Dense)
}
