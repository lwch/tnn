package utils

import "gonum.org/v1/gonum/mat"

func ReshapeRows(input mat.Matrix, rows int) *mat.Dense {
	r, c := input.Dims()
	data := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[i*c+j] = input.At(i, j)
		}
	}
	cols := r * c / rows
	return mat.NewDense(rows, cols, data)
}

func ReshapeCols(input mat.Matrix, cols int) *mat.Dense {
	r, c := input.Dims()
	data := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[i*c+j] = input.At(i, j)
		}
	}
	rows := r * c / cols
	return mat.NewDense(rows, cols, data)
}
