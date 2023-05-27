package math

import (
	"math"

	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
)

// Sigmoid 1 / (1 + exp(-x))
func Sigmoid(x *tensor.Tensor) *tensor.Tensor {
	one := tensor.Ones(x.Dims())
	return x.Scale(-1).Exp().Add(one).Inv()
}

// Softmax exp(x) / sum(exp(max(x)))
func Softmax(x *tensor.Tensor, axis int) *tensor.Tensor {
	max := x.MaxAxis(axis)
	exp := x.Sub(max).Exp()
	return exp.DivElem(exp.SumAxis(axis))
}

// LogSoftmax x - max(x) - log(sum(exp(x - max(x))))
func LogSoftmax(x *tensor.Tensor, axis int) *tensor.Tensor {
	max := x.MaxAxis(axis)
	exp := x.Sub(max).Exp()
	sum := exp.SumAxis(axis)
	return x.Sub(max).Sub(sum.Log())
}

// Mean 按列求均值
func Mean(x *tensor.Tensor) []float64 {
	rows, cols := x.Dims()
	ret := make([]float64, rows)
	for i := 0; i < rows; i++ {
		ret[i] = mat.Sum(x.Value().RowView(i)) / float64(cols)
	}
	return ret
}

// Var 按列求方差
func Var(x *tensor.Tensor) []float64 {
	rows, cols := x.Dims()
	means := Mean(x)
	ret := make([]float64, rows)
	for i := 0; i < rows; i++ {
		mean := means[i]
		for j := 0; j < cols; j++ {
			diff := x.Value().At(i, j) - mean
			ret[i] += math.Pow(diff, 2)
		}
		ret[i] /= float64(cols)
	}
	return ret
}
