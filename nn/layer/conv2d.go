package layer

import (
	"tnn/initializer"

	"gonum.org/v1/gonum/mat"
)

type Conv2D struct {
	*base
}

func NewConv2D(imgShape, kernel, stride Shape,
	init initializer.Initializer) *Conv2D {
	var layer Conv2D
	layer.base = new("conv2d", map[string]Shape{
		"w": {kernel.M, kernel.N},
	}, init, layer.forward, layer.backward)
	return &layer
}

func (layer *Conv2D) forward(input mat.Matrix) mat.Matrix {
	if !layer.hasInit {
		layer.initParams()
	}
	return nil
}

func (layer *Conv2D) backward(grad mat.Matrix) mat.Matrix {
	return nil
}
