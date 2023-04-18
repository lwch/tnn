package layer

import (
	"fmt"

	"github.com/lwch/tnn/initializer"
	"github.com/lwch/tnn/internal/utils"
	"github.com/lwch/tnn/nn/pb"
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
	}, init, layer.forward, layer.backward)
	layer.imageShape = imgShape
	layer.kernelShape = kernel
	layer.stride = stride
	return &layer
}

func LoadConv2D(name string, params map[string]*pb.Dense, args map[string]*pb.Dense) Layer {
	getShape := func(name string) Shape {
		shape := args[name].GetData()
		return Shape{M: int(shape[0]), N: int(shape[1])}
	}
	getStride := func(name string) Stride {
		stride := args[name].GetData()
		return Stride{X: int(stride[0]), Y: int(stride[1])}
	}
	var layer Conv2D
	layer.imageShape = getShape("img.shape")
	layer.kernelShape = getShape("kernel.shape")
	layer.stride = getStride("stride")
	layer.base = new("conv2d", nil, nil, layer.forward, layer.backward)
	layer.name = name
	layer.base.loadParams(params)
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

func (layer *Conv2D) Args() map[string]mat.Matrix {
	buildShape := func(shape Shape) mat.Matrix {
		return mat.NewVecDense(2, []float64{float64(shape.M), float64(shape.N)})
	}
	buildStride := func(stride Stride) mat.Matrix {
		return mat.NewVecDense(2, []float64{float64(stride.X), float64(stride.Y)})
	}
	return map[string]mat.Matrix{
		"img.shape":    buildShape(layer.imageShape),
		"kernel.shape": buildShape(layer.kernelShape),
		"stride":       buildStride(layer.stride),
	}
}

func (layer *Conv2D) Print() {
	layer.base.Print()
	fmt.Println("    Image Shape:",
		fmt.Sprintf("%dx%d", layer.imageShape.M, layer.imageShape.N))
	fmt.Println("    Kernel Shape:",
		fmt.Sprintf("%dx%d", layer.kernelShape.M, layer.kernelShape.N))
	fmt.Println("    Stride:",
		fmt.Sprintf("x=%d", layer.stride.X), fmt.Sprintf("y=%d", layer.stride.Y))
}
