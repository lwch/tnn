package vector

import "gonum.org/v1/gonum/mat"

type Vector3D []*mat.Dense

func Reshape3D(vec mat.Vector, rows, cols int) *Vector3D {
	var ret Vector3D
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
		ret = append(ret, mat.NewDense(rows, cols, tmp))
		i += rows * cols
	}
	return &ret
}

func (v *Vector3D) Get(n int) *mat.Dense {
	return (*v)[n]
}
