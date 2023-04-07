package loss

import "gonum.org/v1/gonum/mat"

type Loss interface {
	Loss(predict, targets *mat.Dense) float64
	Grad(predict, targets *mat.Dense) *mat.Dense
}
