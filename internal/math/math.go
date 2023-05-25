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

func max(x *tensor.Tensor, axis int) *mat.VecDense {
	rows, cols := x.Dims()
	switch axis {
	case 0:
		max := mat.NewVecDense(cols, nil)
		for i := 0; i < rows; i++ {
			if i == 0 {
				max.CopyVec(x.Value().RowView(0))
			}
			for j := 0; j < cols; j++ {
				if max.AtVec(j) < x.Value().At(i, j) {
					max.SetVec(j, x.Value().At(i, j))
				}
			}
		}
		return max
	case 1:
		max := mat.NewVecDense(rows, nil)
		for i := 0; i < rows; i++ {
			if i == 0 {
				max.CopyVec(x.Value().ColView(0))
			}
			for j := 0; j < cols; j++ {
				if max.AtVec(i) < x.Value().At(i, j) {
					max.SetVec(i, x.Value().At(i, j))
				}
			}
		}
		return max
	}
	panic("invalid axis")
}

func sum(x *tensor.Tensor, axis int) *mat.VecDense {
	rows, cols := x.Dims()
	switch axis {
	case 0:
		sum := mat.NewVecDense(cols, nil)
		for i := 0; i < rows; i++ {
			sum.AddVec(sum, x.Value().RowView(i))
		}
		return sum
	case 1:
		sum := mat.NewVecDense(rows, nil)
		for i := 0; i < cols; i++ {
			sum.AddVec(sum, x.Value().ColView(i))
		}
		return sum
	}
	panic("invalid axis")
}

func expand(v *mat.VecDense, rows, cols, axis int) *mat.Dense {
	ret := mat.NewDense(rows, cols, nil)
	switch axis {
	case 0:
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				ret.Set(i, j, v.AtVec(j)+1e-6)
			}
		}
		return ret
	case 1:
		for i := 0; i < rows; i++ {
			n := v.AtVec(i) + 1e-6
			for j := 0; j < cols; j++ {
				ret.Set(i, j, n)
			}
		}
		return ret
	}
	panic("invalid axis")
}

// Softmax exp(x) / sum(exp(max(x,axis)))
func Softmax(x *tensor.Tensor, axis int) *tensor.Tensor {
	max := max(x, axis)
	rows, cols := x.Dims()
	dense := expand(max, rows, cols, axis)
	exp := x.Sub(tensor.FromDense(dense)).Exp()
	sum := sum(exp, axis)
	dense = expand(sum, rows, cols, axis)
	v := exp.MulElem(tensor.FromDense(dense).Inv()).Value()
	return tensor.FromDense(v)
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
