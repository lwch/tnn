package layer

import (
	"tnn/internal/matrix"
	"tnn/internal/shape"
)

type Linear struct {
	w, b *matrix.Matrix
}

func NewLinear(inputShape shape.Shape, outputN int) *Linear {
	return &Linear{
		w: matrix.New(inputShape.M, inputShape.N),
	}
}

func (layer *Linear) Name() string {
	return "linear"
}
