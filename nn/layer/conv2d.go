package layer

import (
	"fmt"

	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/internal/utils"
	"github.com/lwch/tnn/internal/vector"
	"github.com/lwch/tnn/nn/initializer"
	"gonum.org/v1/gonum/mat"
)

type Conv2D struct {
	*base
	padedShape Shape
	imageShape Shape
	kernel     Kernel
	stride     Stride
}

func NewConv2D(imgShape Shape, kernel Kernel, stride Stride,
	init initializer.Initializer) *Conv2D {
	var layer Conv2D
	layer.base = new("conv2d", map[string]Shape{
		"w": {kernel.M * kernel.N * kernel.InChan, kernel.OutChan},
		"b": {imgShape.M * imgShape.N, kernel.OutChan},
	}, init, layer.forward, layer.backward)
	layer.imageShape = imgShape
	layer.kernel = kernel
	layer.stride = stride
	return &layer
}

func LoadConv2D(name string, params map[string]*pb.Dense, args map[string]*pb.Dense) Layer {
	getShape := func(name string) Shape {
		shape := args[name].GetData()
		return Shape{M: int(shape[0]), N: int(shape[1])}
	}
	getKernel := func(name string) Kernel {
		kernel := args[name].GetData()
		return Kernel{M: int(kernel[0]), N: int(kernel[1]),
			InChan: int(kernel[2]), OutChan: int(kernel[3])}
	}
	getStride := func(name string) Stride {
		stride := args[name].GetData()
		return Stride{X: int(stride[0]), Y: int(stride[1])}
	}
	var layer Conv2D
	layer.imageShape = getShape("img.shape")
	layer.kernel = getKernel("kernel")
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
	col := pad.Im2Col(layer.kernel.M, layer.kernel.N,
		layer.stride.Y, layer.stride.X,
		layer.kernel.InChan)
	layer.input.CloneFrom(col)
	var ret mat.Dense
	ret.Mul(col, layer.params["w"])
	ret.Add(&ret, layer.params["b"])
	return utils.ReshapeRows(&ret, batch)
}

func (layer *Conv2D) backward(grad mat.Matrix) mat.Matrix {
	dw := layer.context["w"]
	db := layer.context["b"]

	batch, _ := grad.Dims()
	flatGrad := utils.ReshapeCols(grad, layer.kernel.OutChan)
	dw.(utils.DenseMul).Mul(layer.input.T(), flatGrad)
	db.(utils.DenseCopy).Copy(flatGrad)

	var ret mat.Dense
	w := layer.params["w"]
	ret.Mul(flatGrad, utils.ReshapeCols(w, layer.kernel.OutChan).T())
	ret3D := vector.ReshapeMatrix(&ret, layer.kernel.M, layer.kernel.N)

	tmp := vector.NewVector3D(batch*layer.kernel.InChan, layer.padedShape.M, layer.padedShape.N)
	tmp.ConvAdd(ret3D, layer.stride.Y, layer.stride.X)
	cuted := tmp.Cut(layer.imageShape.M, layer.imageShape.N).ToMatrix()
	return utils.ReshapeRows(cuted, batch)
}

func (layer *Conv2D) pad(input mat.Matrix) *vector.Vector3D {
	reshape := vector.ReshapeMatrix(input, layer.imageShape.M, layer.imageShape.N)
	reshape.Pad(layer.kernel.M-1, layer.kernel.N-1)
	return reshape
}

func (layer *Conv2D) Args() map[string]mat.Matrix {
	buildShape := func(shape Shape) mat.Matrix {
		return mat.NewVecDense(2, []float64{float64(shape.M), float64(shape.N)})
	}
	buildKernel := func(kernel Kernel) mat.Matrix {
		return mat.NewVecDense(4, []float64{
			float64(kernel.M), float64(kernel.N),
			float64(kernel.InChan), float64(kernel.OutChan)})
	}
	buildStride := func(stride Stride) mat.Matrix {
		return mat.NewVecDense(2, []float64{float64(stride.X), float64(stride.Y)})
	}
	return map[string]mat.Matrix{
		"img.shape": buildShape(layer.imageShape),
		"kernel":    buildKernel(layer.kernel),
		"stride":    buildStride(layer.stride),
	}
}

func (layer *Conv2D) Print() {
	layer.base.Print()
	fmt.Println("    Image Shape:",
		fmt.Sprintf("%dx%d", layer.imageShape.M, layer.imageShape.N))
	fmt.Println("    Kernel:",
		fmt.Sprintf("%dx%d", layer.kernel.M, layer.kernel.N),
		fmt.Sprintf("input_channel=%d", layer.kernel.InChan),
		fmt.Sprintf("output_channel=%d", layer.kernel.OutChan))
	fmt.Println("    Stride:",
		fmt.Sprintf("x=%d", layer.stride.X), fmt.Sprintf("y=%d", layer.stride.Y))
}
