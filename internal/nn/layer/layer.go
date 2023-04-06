package layer

import "tnn/internal/matrix"

type Layer interface {
	Name() string
	Forward(input *matrix.Matrix) *matrix.Matrix
	Backward(grad *matrix.Matrix) *matrix.Matrix
}
