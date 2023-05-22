package tensor

import (
	"gonum.org/v1/gonum/mat"
)

func Ones(rows, cols int) *Tensor {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = 1
	}
	return New(data, rows, cols)
}

func Zeros(rows, cols int) *Tensor {
	data := make([]float64, rows*cols)
	return New(data, rows, cols)
}

func Numbers(rows, cols int, n float64) *Tensor {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = n
	}
	return New(data, rows, cols)
}

func vec2Dense(vector *mat.VecDense) *mat.Dense {
	cols, _ := vector.Dims()
	data := make([]float64, cols)
	for i := range data {
		data[i] = vector.AtVec(i)
	}
	return mat.NewDense(1, cols, data)
}

func vecRepeat(vector *mat.VecDense, rows int) *mat.Dense {
	cols, _ := vector.Dims()
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = vector.AtVec(i % cols)
	}
	return mat.NewDense(rows, cols, data)
}
