package vector

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Vector3D struct {
	rows, cols int
	data       []mat.Matrix
}

func Reshape3D(vec mat.Vector, rows, cols int) *Vector3D {
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
	offsetY := m >> 1
	offsetX := n >> 1
	for i := 0; i < v.Size(); i++ {
		dense := mat.NewDense(v.rows+offsetY<<1, v.cols+offsetX<<1, nil)
		for j := 0; j < v.rows; j++ {
			row := v.data[i].(RowViewer).RowView(j)
			for k := 0; k < row.Len(); k++ {
				dense.Set(j+offsetY, k+offsetX, row.AtVec(k))
			}
		}
		v.data[i] = dense
	}
	v.rows += offsetY << 1
	v.cols += offsetX << 1
}

func (v *Vector3D) Conv(kernel mat.Matrix, strideY, strideX int) *Vector3D {
	var ret Vector3D
	kernelM, kernelN := kernel.Dims()
	dm := float64(v.rows - kernelM)
	dn := float64(v.cols - kernelN)
	ret.rows = int(math.Ceil(dm / float64(strideY)))
	ret.cols = int(math.Ceil(dn / float64(strideX)))
	ret.data = make([]mat.Matrix, v.Size())
	for i := 0; i < v.Size(); i++ {
		dense := mat.NewDense(ret.rows, ret.cols, nil)
		for row := 0; row < v.rows-kernelM; row++ {
			for col := 0; col < v.cols-kernelN; col++ {
				a := v.data[i].(Slicer).Slice(row, row+kernelM, col, col+kernelN)
				var tmp mat.Dense
				tmp.MulElem(a, kernel)
				dense.Set(row, col, mat.Sum(&tmp))
			}
		}
		ret.data[i] = dense
	}
	return &ret
}

func (v *Vector3D) Add(bias mat.Matrix) {
	for i := 0; i < v.Size(); i++ {
		dense := v.data[i]
		dense.(Adder).Add(dense, bias)
		v.data[i] = dense
	}
}
