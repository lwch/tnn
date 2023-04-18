package layer

import (
	"github.com/lwch/tnn/initializer"
	"github.com/lwch/tnn/internal/utils"
	"github.com/lwch/tnn/nn/vector"
	"gonum.org/v1/gonum/mat"
)

type Conv2D struct {
	*base
	padedShape              Shape
	imageShape, kernelShape Shape
	stride                  Stride
}

func NewConv2D(imgShape, kernel Shape, stride Stride,
	init initializer.Initializer) *Conv2D {
	var layer Conv2D
	layer.base = new("conv2d", map[string]Shape{
		"w": {kernel.M * kernel.N, 1},
		"b": {imgShape.M * imgShape.N, 1},
		// "img":    {1, 2},
		// "kernel": {1, 2},
		// "stride": {1, 2},
	}, init, layer.forward, layer.backward)
	layer.imageShape = imgShape
	layer.kernelShape = kernel
	layer.stride = stride
	return &layer
}

func (layer *Conv2D) OutputShape() Shape {
	return layer.imageShape
}

func (layer *Conv2D) forward(input mat.Matrix) mat.Matrix {
	batch, _ := input.Dims()
	if !layer.hasInit {
		shape := layer.shapes["b"]
		shape.M *= batch
		layer.shapes["b"] = shape
		layer.initParams()
		buildShape := func(shape Shape) mat.Matrix {
			var data [2]float64
			data[0] = float64(shape.M)
			data[1] = float64(shape.N)
			return mat.NewVecDense(2, data[:])
		}
		buildStride := func(stride Stride) mat.Matrix {
			var data [2]float64
			data[0] = float64(stride.Y)
			data[1] = float64(stride.X)
			return mat.NewVecDense(2, data[:])
		}
		layer.params["img"] = buildShape(layer.imageShape)
		layer.params["kernel"] = buildShape(layer.kernelShape)
		layer.params["stride"] = buildStride(layer.stride)
	}
	pad := layer.pad(input)
	layer.padedShape.M, layer.padedShape.N = pad.Dims()
	col := pad.Im2Col(layer.kernelShape.M, layer.kernelShape.N, layer.stride.Y, layer.stride.X)
	layer.input = *col
	var ret mat.Dense
	ret.Mul(col, layer.params["w"])
	ret.Add(&ret, layer.params["b"])
	return utils.ReshapeRows(&ret, batch)
}

func (layer *Conv2D) backward(grad mat.Matrix) mat.Matrix {
	dw := layer.context["w"]
	db := layer.context["b"]

	rows, cols := grad.Dims()
	flatGrad := utils.ReshapeRows(grad, rows*cols)
	dw.(vector.Muler).Mul(layer.input.T(), flatGrad)
	db.(vector.Copyer).Copy(flatGrad)

	var ret mat.Dense
	w := layer.params["w"]
	tGrad := utils.ReshapeRows(grad.T(), rows*cols)
	ret.Mul(tGrad, w.T())
	ret3D := vector.ReshapeMatrix(&ret, layer.kernelShape.M, layer.kernelShape.N)

	tmp := vector.NewVector3D(rows, layer.padedShape.M, layer.padedShape.N)
	tmp.ConvAdd(ret3D, layer.stride.Y, layer.stride.X)
	return tmp.Cut(layer.imageShape.M, layer.imageShape.N).ToMatrix()
}

func (layer *Conv2D) pad(input mat.Matrix) *vector.Vector3D {
	reshape := vector.ReshapeMatrix(input, layer.imageShape.M, layer.imageShape.N)
	reshape.Pad(layer.kernelShape.M-1, layer.kernelShape.N-1)
	return reshape
}
