package layer

import (
	"fmt"
	"math"

	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/internal/utils"
	"github.com/lwch/tnn/internal/vector"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/params"
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
		"b": {1, kernel.OutChan},
	}, init)
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
	layer.base = new("conv2d", nil, nil)
	layer.name = name
	layer.base.loadParams(params)
	return &layer
}

func (layer *Conv2D) OutputShape() Shape {
	dy := float64(layer.imageShape.M - layer.kernel.M)
	dx := float64(layer.imageShape.N - layer.kernel.N)
	y := math.Ceil(dy/float64(layer.stride.Y)) + 1
	x := math.Ceil(dx/float64(layer.stride.X)) + 1
	return Shape{int(y), int(x)}
}

func (layer *Conv2D) KernelShape() Shape {
	return Shape{layer.kernel.M, layer.kernel.N}
}

func (layer *Conv2D) InputChan() int {
	return layer.kernel.InChan
}

func (layer *Conv2D) OutputChan() int {
	return layer.kernel.OutChan
}

// input:  [batch, w*h*inChan]
// output: [batch, w*h*outChan]
func (layer *Conv2D) Forward(input mat.Matrix, _ bool) (context []mat.Matrix, output mat.Matrix) {
	batch, _ := input.Dims()
	if !layer.hasInit {
		layer.initParams()
	}
	// pad right and bottom on each channel, output shape is (input.M+kernel.M-1, input.N+kernel.N-1)
	// input: | 1 2 3 |   kernel: | 1 2 |
	//  (3x3) | 4 5 6 |    (2x2)  | 3 4 |
	//        | 7 8 9 |
	// padding output: (4x4)
	// | 1 2 3 0 |
	// | 4 5 6 0 |
	// | 7 8 9 0 |
	// | 0 0 0 0 |
	// this example is only for input channel is 1
	pad := layer.pad(input)
	layer.padedShape.M, layer.padedShape.N = pad.Dims()
	// get convolution rectangle, output shape is (input.M*input.N, kernel.M*kernel.N*kernel.InChan)
	// input: | 1 2 3 0 |   kernel: | 1 2 |
	//  (4x4) | 4 5 6 0 |    (2x2)  | 3 4 |
	//        | 7 8 9 0 |
	//        | 0 0 0 0 |
	// output list:
	// | 1 2 |  | 2 3 |  | 3 0 |
	// | 3 4 |  | 5 6 |  | 6 0 |
	//
	// | 4 5 |  | 5 6 |  | 6 0 |
	// | 7 8 |  | 8 9 |  | 9 0 |
	//
	// | 7 8 |  | 8 9 |  | 9 0 |
	// | 0 0 |  | 0 0 |  | 0 0 |
	// this example is only for input channel is 1, stride is (1, 1)
	col := pad.Im2Col(layer.kernel.M, layer.kernel.N,
		layer.stride.Y, layer.stride.X,
		layer.kernel.InChan)
	var ctx mat.Dense
	ctx.CloneFrom(col)
	var ret mat.Dense
	ret.Mul(col, layer.params.Get("w"))
	b := layer.params.Get("b").(utils.DenseRowView).RowView(0)
	rows, _ := ret.Dims()
	for i := 0; i < rows; i++ {
		row := ret.RowView(i)
		row.(utils.AddVec).AddVec(row, b)
	}
	return []mat.Matrix{&ctx}, utils.ReshapeRows(&ret, batch)
}

func (layer *Conv2D) Backward(context []mat.Matrix, grad mat.Matrix) (valueGrad mat.Matrix, paramsGrad *params.Params) {
	paramsGrad = params.New()
	dw := paramsGrad.Init("w", layer.shapes["w"].M, layer.shapes["w"].N)
	db := paramsGrad.Init("b", layer.shapes["b"].M, layer.shapes["b"].N)

	// same as dense layer
	batch, _ := grad.Dims()
	flatGrad := utils.ReshapeCols(grad, layer.kernel.OutChan)
	dw.(utils.DenseMul).Mul(context[0].T(), flatGrad)
	db0 := db.(utils.DenseRowView).RowView(0)
	rows, _ := flatGrad.Dims()
	for i := 0; i < rows; i++ {
		db0.(utils.AddVec).AddVec(db0, flatGrad.RowView(i))
	}
	db0.(utils.ScaleVec).ScaleVec(1/float64(rows), db0)

	var ret mat.Dense
	w := layer.params.Get("w")
	ret.Mul(flatGrad, utils.ReshapeCols(w, layer.kernel.OutChan).T())
	ret3D := vector.ReshapeMatrix(&ret, layer.kernel.M, layer.kernel.N)

	// the output shape is same of input shape
	tmp := vector.NewVector3D(batch*layer.kernel.InChan, layer.padedShape.M, layer.padedShape.N)
	tmp.ConvAdd(ret3D, layer.stride.Y, layer.stride.X)
	// cut the padding
	cuted := tmp.Cut(layer.imageShape.M, layer.imageShape.N).ToMatrix()
	return utils.ReshapeRows(cuted, batch), paramsGrad
}

func (layer *Conv2D) pad(input mat.Matrix) *vector.Vector3D {
	padY := layer.stride.Y - (layer.imageShape.M % layer.stride.Y)
	padX := layer.stride.X - (layer.imageShape.N % layer.stride.X)
	reshape := vector.ReshapeMatrix(input, layer.imageShape.M, layer.imageShape.N)
	if padY != layer.stride.Y || padX != layer.stride.X {
		reshape.Pad(padY, padX)
	}
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
