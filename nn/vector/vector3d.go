package vector

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Vector3D struct {
	rows, cols int
	data       []mat.Matrix
}

func NewVector3D(rows, cols int) *Vector3D {
	var ret Vector3D
	ret.rows = rows
	ret.cols = cols
	ret.data = append(ret.data, mat.NewDense(rows, cols, nil))
	return &ret
}

func ReshapeVector(vec mat.Vector, rows, cols int) *Vector3D {
	var ret Vector3D
	ret.rows = rows
	ret.cols = cols
	var i int
	for {
		if i >= vec.Len() {
			break
		}
		var n int
		tmp := make([]float64, rows*cols)
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				tmp[n] = vec.AtVec(i + n)
				n++
			}
		}
		ret.data = append(ret.data, mat.NewDense(rows, cols, tmp))
		i += rows * cols
	}
	return &ret
}

func ReshapeMatrix(d mat.Matrix, rows, cols int) *Vector3D {
	srcRows, srcCols := d.Dims()
	data := make([]float64, srcRows*srcCols)
	idx := 0
	for i := 0; i < srcRows; i++ {
		for j := 0; j < srcCols; j++ {
			data[idx] = d.At(i, j)
			idx++
		}
	}
	return ReshapeVector(mat.NewVecDense(srcRows*srcCols, data), rows, cols)
}

func (v *Vector3D) Get(n int) mat.Matrix {
	return v.data[n]
}

func (v *Vector3D) Size() int {
	return len(v.data)
}

func (v *Vector3D) ToMatrix() mat.Matrix {
	size := v.Size() * v.rows * v.cols
	raw := make([]float64, size)
	for i := 0; i < v.Size(); i++ {
		for j := 0; j < v.rows; j++ {
			for k := 0; k < v.cols; k++ {
				raw[i*v.rows*v.cols+j*v.cols+k] = v.data[i].At(j, k)
			}
		}
	}
	return mat.NewVecDense(size, raw)
}

func (v *Vector3D) Pad(m, n int) {
	for i := 0; i < v.Size(); i++ {
		dense := mat.NewDense(v.rows+m, v.cols+n, nil)
		for j := 0; j < v.rows; j++ {
			row := v.data[i].(RowViewer).RowView(j)
			for k := 0; k < row.Len(); k++ {
				dense.Set(j, k, row.AtVec(k))
			}
		}
		v.data[i] = dense
	}
	v.rows += m
	v.cols += n
}

func (v *Vector3D) Im2Col(kernelM, kernelN, strideM, strideN int) *mat.Dense {
	rows := math.Ceil(float64(v.rows-kernelM)/float64(strideM)) + 1
	cols := math.Ceil(float64(v.cols-kernelN)/float64(strideN)) + 1
	data := make([]float64, v.Size()*int(rows)*int(cols)*kernelM*kernelN)
	copy := func(dst []float64, rect mat.Matrix) {
		rows, cols := rect.Dims()
		for i := 0; i < rows; i++ {
			row := rect.(RowViewer).RowView(i)
			for j := 0; j < cols; j++ {
				dst[i*cols+j] = row.AtVec(j)
			}
		}
	}
	idx := 0
	for i := 0; i < v.Size(); i++ {
		for j := 0; j < int(rows); j++ {
			topLeftY := j * strideM
			bottomRightY := topLeftY + kernelM
			for k := 0; k < int(cols); k++ {
				topLeftX := k * strideN
				bottomRightX := topLeftX + kernelN
				rect := v.data[i].(Slicer).Slice(topLeftY, bottomRightY, topLeftX, bottomRightX)
				copy(data[idx:], rect)
				idx += kernelM * kernelN
			}
		}
	}
	return mat.NewDense(v.Size()*int(rows)*int(cols), kernelM*kernelN, data)
}

func (v *Vector3D) ConvAdd(a *Vector3D, strideM, strideN int) {
	kernelM, kernelN := a.Dims()
	rows := math.Ceil(float64(v.rows-kernelM)/float64(strideM)) + 1
	cols := math.Ceil(float64(v.cols-kernelN)/float64(strideN)) + 1
	idx := 0
	for i := 0; i < int(rows); i += strideM {
		for j := 0; j < int(cols); j += strideN {
			rect := v.data[0].(Slicer).Slice(i, i+a.rows, j, j+a.cols)
			rect.(Adder).Add(rect, a.data[idx])
			idx++
		}
	}
}

func (v *Vector3D) Cut(rows, cols int) *Vector3D {
	var ret Vector3D
	ret.rows = rows
	ret.cols = cols
	ret.data = make([]mat.Matrix, v.Size())
	for i := 0; i < v.Size(); i++ {
		rect := v.data[i].(Slicer).Slice(0, rows, 0, cols)
		ret.data[i] = mat.DenseCopyOf(rect)
	}
	return &ret
}

func (v *Vector3D) Dims() (int, int) {
	return v.rows, v.cols
}
