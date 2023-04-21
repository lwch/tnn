package utils

import (
	"gonum.org/v1/gonum/mat"
)

func ReshapeRows(input mat.Matrix, rows int) *mat.Dense {
	r, c := input.Dims()
	cols := r * c / rows
	switch dense := input.(type) {
	case *mat.Dense:
		return mat.NewDense(rows, cols, dense.RawMatrix().Data)
	case *mat.VecDense:
		return mat.NewDense(rows, cols, dense.RawVector().Data)
	default:
		data := make([]float64, r*c)
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				data[i*c+j] = input.At(i, j)
			}
		}
		return mat.NewDense(rows, cols, data)
	}
}

func ReshapeCols(input mat.Matrix, cols int) *mat.Dense {
	r, c := input.Dims()
	rows := r * c / cols
	switch dense := input.(type) {
	case *mat.Dense:
		return mat.NewDense(rows, cols, dense.RawMatrix().Data)
	case *mat.VecDense:
		return mat.NewDense(rows, cols, dense.RawVector().Data)
	default:
		data := make([]float64, r*c)
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				data[i*c+j] = input.At(i, j)
			}
		}
		return mat.NewDense(rows, cols, data)
	}
}
