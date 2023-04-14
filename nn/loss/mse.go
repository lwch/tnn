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

func (*MSE) Loss(predict mat.Matrix, targets *mat.Dense) float64 {
	var tmp mat.Dense
	var sum float64
	tmp.Apply(func(i, j int, v float64) float64 {
		diff := v - targets.At(i, j)
		sum += math.Pow(diff, 2)
		return 0
	}, predict)
	rows, _ := predict.Dims()
	return 0.5 * sum / float64(rows)
}

func (*MSE) Grad(predict mat.Matrix, targets *mat.Dense) *mat.Dense {
	var grad mat.Dense
	grad.Sub(predict, targets)
	rows, _ := predict.Dims()
	grad.Scale(1/float64(rows), &grad)
	return &grad
}

func (loss *MSE) Save() *pb.Loss {
	return &pb.Loss{Name: "mse"}
}
