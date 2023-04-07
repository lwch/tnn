package layer

import (
	"math"
	"tnn/internal/nn/optimizer"

	"gonum.org/v1/gonum/mat"
)

type Sigmoid struct {
	input mat.Dense
}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (layer *Sigmoid) Name() string {
	return "sigmoid"
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (layer *Sigmoid) Forward(input *mat.Dense) *mat.Dense {
	layer.input.CloneFrom(input)
	rows, cols := input.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			layer.input.Set(i, j, sigmoid(input.At(i, j)))
		}
	}
	return &layer.input
}

func (layer *Sigmoid) Backward(grad *mat.Dense) *mat.Dense {
	rows, cols := layer.input.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			n := layer.input.At(i, j)
			layer.input.Set(i, j, n*(1-n)*grad.At(i, j))
		}
	}
	return &layer.input
}

func (layer *Sigmoid) Update(optimizer optimizer.Optimizer) {
}
