package loss

import (
	"math"
	"tnn/nn/pb"

	"gonum.org/v1/gonum/mat"
)

type MSE struct{}

func NewMSE() *MSE {
	return &MSE{}
}

func (*MSE) Name() string {
	return "mse"
}

func (*MSE) Loss(predict, targets *mat.Dense) float64 {
	row, col := predict.Dims()
	var sum float64
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			diff := predict.At(i, j) - targets.At(i, j)
			sum += math.Pow(diff, 2)
		}
	}
	return 0.5 * sum / float64(row)
}

func (*MSE) Grad(predict, targets *mat.Dense) *mat.Dense {
	var grad mat.Dense
	grad.Sub(predict, targets)
	rows, _ := predict.Dims()
	grad.Scale(1/float64(rows), &grad)
	return &grad
}

func (loss *MSE) Save() *pb.Loss {
	return &pb.Loss{Name: "mse"}
}
