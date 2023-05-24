package math

import (
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
			ret.RowView(i).(*mat.VecDense).CopyVec(v)
		}
		return ret
	case 1:
		for i := 0; i < cols; i++ {
			ret.ColView(i).(*mat.VecDense).CopyVec(v)
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
	return exp.MulElem(tensor.FromDense(dense).Inv())
}
