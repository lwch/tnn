package vector

import (
	"math"
	"sync"

	"github.com/lwch/tnn/internal/utils"
	"gonum.org/v1/gonum/mat"
)

type Vector3D struct {
	rows, cols int
	data       []mat.Matrix
}

func NewVector3D(batch, rows, cols int) *Vector3D {
	var ret Vector3D
	ret.rows = rows
	ret.cols = cols
	for i := 0; i < batch; i++ {
		ret.data = append(ret.data, mat.NewDense(rows, cols, nil))
	}
	return &ret
}

func ReshapeVector(vec mat.Vector, rows, cols int) *Vector3D {
	var ret Vector3D
	ret.rows = rows
	ret.cols = cols
	size := rows * cols
	data := vec.(utils.RawVector).RawVector().Data
	ret.data = make([]mat.Matrix, 0, vec.Len()/size)
	for i := 0; i < vec.Len(); i += size {
		tmp := make([]float64, size)
		copy(tmp, data[i:i+size])
		ret.data = append(ret.data, mat.NewDense(rows, cols, tmp))
	}
	return &ret
}

func ReshapeMatrix(d mat.Matrix, rows, cols int) *Vector3D {
	srcRows, srcCols := d.Dims()
	data := make([]float64, srcRows*srcCols)
	for row := 0; row < srcRows; row++ {
		for col := 0; col < srcCols; col++ {
			data[row*srcCols+col] = d.At(row, col)
		}
	}
	return ReshapeVector(mat.NewVecDense(srcRows*srcCols, data), rows, cols)
}

func (v *Vector3D) Get(batch int) mat.Matrix {
	return v.data[batch]
}

func (v *Vector3D) BatchSize() int {
	return len(v.data)
}

func (v *Vector3D) ToMatrix() mat.Matrix {
	raw := make([]float64, v.BatchSize()*v.rows*v.cols)
	size := v.rows * v.cols
	for i := 0; i < v.BatchSize(); i++ {
		start := i * size
		copy(raw[start:start+size], v.data[i].(*mat.Dense).RawMatrix().Data)
	}
	return mat.NewDense(v.BatchSize(), size, raw)
}

// Pad padding matrix on right and bottom
func (v *Vector3D) Pad(m, n int) {
	for i := 0; i < v.BatchSize(); i++ {
		dense := mat.NewDense(v.rows+m, v.cols+n, nil)
		for j := 0; j < v.rows; j++ {
			row := v.data[i].(utils.DenseRowView).RowView(j)
			for k := 0; k < row.Len(); k++ {
				dense.Set(j, k, row.AtVec(k))
			}
		}
		v.data[i] = dense
	}
	v.rows += m
	v.cols += n
}

// Im2Col get convolution matrix, output shape is (batch * conved.rows * conved.cols, kernelM * kernelN * channelSize)
func (v *Vector3D) Im2Col(kernelM, kernelN, strideM, strideN, channelSize int) *mat.Dense {
	rows := math.Ceil(float64(v.rows-kernelM)/float64(strideM)) + 1
	cols := math.Ceil(float64(v.cols-kernelN)/float64(strideN)) + 1
	batch := v.BatchSize() / channelSize
	ret := mat.NewDense(batch*int(rows)*int(cols), kernelM*kernelN*channelSize, nil)
	copy := func(dst []float64, rect mat.Matrix) {
		rows, cols := rect.Dims()
		for i := 0; i < rows; i++ {
			row := rect.(utils.DenseRowView).RowView(i)
			idx := i * cols
			copy(dst[idx:idx+row.Len()], row.(*mat.VecDense).RawVector().Data)
		}
	}
	var wg sync.WaitGroup
	for i := 0; i < batch; i++ {
		offset := i * int(rows) * int(cols)
		for j := 0; j < int(rows); j++ {
			topLeftY := j * strideM
			bottomRightY := topLeftY + kernelM
			for k := 0; k < int(cols); k++ {
				topLeftX := k * strideN
				bottomRightX := topLeftX + kernelN
				row := ret.RowView(offset + j*int(cols) + k)
				data := row.(*mat.VecDense).RawVector().Data
				for channel := 0; channel < channelSize; channel++ {
					rect := v.data[i*channelSize+channel].(utils.DenseSlice).
						Slice(topLeftY, bottomRightY, topLeftX, bottomRightX)
					wg.Add(1)
					go func(channel int, rect mat.Matrix) {
						defer wg.Done()
						copy(data[channel*kernelM*kernelN:], rect)
					}(channel, rect)
				}
			}
		}
	}
	wg.Wait()
	return ret
}

// ConvAdd add gradient to each channel
func (v *Vector3D) ConvAdd(a *Vector3D, strideM, strideN int) {
	kernelM, kernelN := a.Dims()
	rows := math.Ceil(float64(v.rows-kernelM)/float64(strideM)) + 1
	cols := math.Ceil(float64(v.cols-kernelN)/float64(strideN)) + 1
	idx := 0
	for batch := 0; batch < v.BatchSize(); batch++ {
		for i := 0; i < int(rows); i += strideM {
			for j := 0; j < int(cols); j += strideN {
				rect := v.data[batch].(utils.DenseSlice).Slice(i, i+a.rows, j, j+a.cols)
				rect.(utils.DenseAdd).Add(rect, a.data[idx])
				idx++
			}
		}
	}
}

// Cut cut matrix to (rows, cols)
func (v *Vector3D) Cut(rows, cols int) *Vector3D {
	var ret Vector3D
	ret.rows = rows
	ret.cols = cols
	ret.data = make([]mat.Matrix, v.BatchSize())
	for i := 0; i < v.BatchSize(); i++ {
		rect := v.data[i].(utils.DenseSlice).Slice(0, rows, 0, cols)
		ret.data[i] = mat.DenseCopyOf(rect)
	}
	return &ret
}

func (v *Vector3D) Dims() (int, int) {
	return v.rows, v.cols
}
