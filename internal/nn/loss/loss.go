package loss

import (
	"tnn/internal/nn/pb"

	"gonum.org/v1/gonum/mat"
)

type Loss interface {
	Loss(predict, targets *mat.Dense) float64
	Grad(predict, targets *mat.Dense) *mat.Dense
	Save() *pb.Loss
}
