package layer

import (
	"tnn/internal/nn/optimizer"

	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	Name() string
	Forward(input *mat.Dense) *mat.Dense
	Backward(grad *mat.Dense) *mat.Dense
	Update(optimizer optimizer.Optimizer)
}
