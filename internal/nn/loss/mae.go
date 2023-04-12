package loss

import (
	"math"
	"tnn/internal/nn/pb"

	"gonum.org/v1/gonum/mat"
)

type MAE struct{}

func NewMAE() *MAE {
	return &MAE{}
}

func (*MAE) Loss(predict, targets *mat.Dense) float64 {
	row, col := predict.Dims()
	var sum float64
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			sum += math.Abs(predict.At(i, j) - targets.At(i, j))
		}
	}
	return sum / float64(row)
}

func (*MAE) Grad(predict, targets *mat.Dense) *mat.Dense {
	rows, cols := predict.Dims()
	grad := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			n := predict.At(i, j) - targets.At(i, j)
			var sign float64
			if n > 0 {
				sign = 1
			} else if n < 0 {
				sign = -1
			}
			grad.Set(i, j, sign/float64(rows))
		}
	}
	return grad
}

func (loss *MAE) Save() *pb.Loss {
	return &pb.Loss{Name: "mae"}
}
