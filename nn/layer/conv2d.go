package layer

import (
	"tnn/initializer"
	"tnn/nn/vector"

	"gonum.org/v1/gonum/mat"
)

type Conv2D struct {
	*base
	imageShape, kernelShape Shape
	stride                  Stride
}

func NewConv2D(imgShape, kernel Shape, stride Stride,
	init initializer.Initializer) *Conv2D {
	var layer Conv2D
	layer.base = new("conv2d", map[string]Shape{
		"w":      {kernel.M, kernel.N},
		"b":      {imgShape.M, imgShape.N},
		"img":    {1, 2},
		"kernel": {1, 2},
		"stride": {1, 2},
	}, init, layer.forward, layer.backward)
	layer.imageShape = imgShape
	layer.kernelShape = kernel
	layer.stride = stride
	return &layer
}

func (layer *Conv2D) forward(input mat.Matrix) mat.Matrix {
	if !layer.hasInit {
		layer.initParams()
		buildShape := func(shape Shape) mat.Matrix {
			var data [2]float64
			data[0] = float64(shape.M)
			data[1] = float64(shape.N)
			return mat.NewDense(1, 2, data[:])
		}
		buildStride := func(stride Stride) mat.Matrix {
			var data [2]float64
			data[0] = float64(stride.Y)
			data[1] = float64(stride.X)
			return mat.NewDense(1, 2, data[:])
		}
		layer.params["img"] = buildShape(layer.imageShape)
		layer.params["kernel"] = buildShape(layer.kernelShape)
		layer.params["stride"] = buildStride(layer.stride)
	}
	reshape := layer.pad(input)
	ret := reshape.Conv(layer.params["w"], layer.stride.Y, layer.stride.X)
	ret.Add(layer.params["b"])
	return ret.ToMatrix()
}

func (layer *Conv2D) backward(grad mat.Matrix) mat.Matrix {
	return nil
}

func (layer *Conv2D) pad(input mat.Matrix) *vector.Vector3D {
	reshape := vector.Reshape3D(input.(mat.Vector), layer.imageShape.M, layer.imageShape.N)
	reshape.Pad(layer.kernelShape.M, layer.kernelShape.N)
	return reshape
}
