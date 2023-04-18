package layer

import (
	"fmt"
	"math"

	"github.com/lwch/tnn/internal/utils"
	"github.com/lwch/tnn/nn/pb"
	"github.com/lwch/tnn/nn/vector"
	"gonum.org/v1/gonum/mat"
)

type MaxPool struct {
	*base
	padedShape Shape
	imageShape Shape
	kernel     Kernel
	stride     Stride
	idx        [][][]int
}

func NewMaxPool(imgShape Shape, kernel Kernel, stride Stride) *MaxPool {
	var layer MaxPool
	layer.base = new("maxpool", nil, nil, layer.forward, layer.backward)
	layer.imageShape = imgShape
	layer.kernel = kernel
	layer.stride = stride
	return &layer
}

func LoadMaxPool(name string, params map[string]*pb.Dense, args map[string]*pb.Dense) Layer {
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
	var layer MaxPool
	layer.imageShape = getShape("img.shape")
	layer.kernel = getKernel("kernel")
	layer.stride = getStride("stride")
	layer.base = new("maxpool", nil, nil, layer.forward, layer.backward)
	layer.name = name
	layer.base.loadParams(params)
	batch := int(args["batch"].GetData()[0])
	layer.idx = make([][][]int, batch)
	outputShape := layer.OutputShape()
	for i := 0; i < batch; i++ {
		layer.idx[i] = make([][]int, layer.kernel.InChan)
		for j := 0; j < layer.kernel.InChan; j++ {
			layer.idx[i][j] = make([]int, outputShape.M*outputShape.N)
		}
	}
	return &layer
}

func (layer *MaxPool) OutputShape() Shape {
	dy := float64(layer.imageShape.M - layer.kernel.M)
	dx := float64(layer.imageShape.N - layer.kernel.N)
	y := math.Ceil(dy/float64(layer.stride.Y)) + 1
	x := math.Ceil(dx/float64(layer.stride.X)) + 1
	return Shape{int(y), int(x)}
}

func (layer *MaxPool) forward(input mat.Matrix) mat.Matrix {
	batch, _ := input.Dims()
	outputShape := layer.OutputShape()
	if !layer.hasInit {
		layer.initParams()
		layer.idx = make([][][]int, batch)
		for i := 0; i < batch; i++ {
			layer.idx[i] = make([][]int, layer.kernel.InChan)
			for j := 0; j < layer.kernel.InChan; j++ {
				layer.idx[i][j] = make([]int, outputShape.M*outputShape.N)
			}
		}
	}
	pad := layer.pad(input)
	layer.padedShape.M, layer.padedShape.N = pad.Dims()
	maxFunc := func(m mat.Matrix) (float64, int) {
		rows, cols := m.Dims()
		max := math.Inf(-1)
		idx := -1
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				v := m.At(i, j)
				if v > max {
					max = v
					idx = i*cols + j
				}
			}
		}
		return max, idx
	}
	ret := mat.NewDense(batch, outputShape.M*outputShape.N*layer.kernel.InChan, nil)
	var layerID int
	for i := 0; i < pad.Size(); i++ {
		batchID := math.Floor(float64(i) / float64(layer.kernel.InChan))
		img := pad.Get(i)
		if i%layer.kernel.InChan == 0 {
			layerID = 0
		}
		var idx int
		for j := 0; j < outputShape.M; j++ {
			topLeftY := j * layer.stride.Y
			bottomRightY := topLeftY + layer.kernel.M
			for k := 0; k < outputShape.N; k++ {
				topLeftX := k * layer.stride.X
				bottomRightX := topLeftX + layer.kernel.N
				rect := img.(vector.Slicer).Slice(topLeftY, bottomRightY, topLeftX, bottomRightX)
				var value float64
				value, layer.idx[int(batchID)][layerID][idx] = maxFunc(rect)
				ret.Set(int(batchID), idx, value)
				idx++
			}
		}
		layerID++
	}
	return ret
}

func (layer *MaxPool) backward(grad mat.Matrix) mat.Matrix {
	batch, _ := grad.Dims()
	ret := vector.NewVector3D(batch*layer.kernel.InChan, layer.padedShape.M, layer.padedShape.N)
	outputShape := layer.OutputShape()
	for i := 0; i < batch; i++ {
		rv := grad.(vector.RowViewer).RowView(i)
		var gradIdx int
		for layerID := 0; layerID < layer.kernel.InChan; layerID++ {
			img := ret.Get(i*layer.kernel.InChan + layerID)
			var idx int
			for j := 0; j < outputShape.M; j++ {
				startY := j * layer.stride.Y
				for k := 0; k < outputShape.N; k++ {
					startX := k * layer.stride.X
					g := rv.AtVec(gradIdx)
					n := layer.idx[i][layerID][idx]
					dy := math.Floor(float64(n) / float64(layer.kernel.N))
					dx := n % layer.kernel.N
					g += img.At(startY+int(dy), startX+dx)
					img.(vector.Seter).Set(startY+int(dy), startX+dx, g)
					idx++
					gradIdx++
				}
			}
		}
	}
	tmp := ret.Cut(layer.imageShape.M, layer.imageShape.N).ToMatrix()
	return utils.ReshapeRows(tmp, batch)
}

func (layer *MaxPool) pad(input mat.Matrix) *vector.Vector3D {
	padY := layer.stride.Y - (layer.imageShape.M % layer.stride.Y)
	padX := layer.stride.X - (layer.imageShape.N % layer.stride.X)
	reshape := vector.ReshapeMatrix(input, layer.imageShape.M, layer.imageShape.N)
	if padY != layer.stride.Y || padX != layer.stride.X {
		reshape.Pad(padY, padX)
	}
	return reshape
}

func (layer *MaxPool) Args() map[string]mat.Matrix {
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
		"img.shape":    buildShape(layer.imageShape),
		"kernel.shape": buildKernel(layer.kernel),
		"stride":       buildStride(layer.stride),
		"batch":        mat.NewVecDense(1, []float64{float64(len(layer.idx))}),
	}
}

func (layer *MaxPool) Print() {
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
