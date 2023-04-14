package loss

import (
	"math"
	"tnn/nn/pb"

	"gonum.org/v1/gonum/mat"
)

type MAE struct{}

func NewMAE() *MAE {
	return &MAE{}
}

func (*MAE) Name() string {
	return "mae"
}

func (*MAE) Loss(predict mat.Matrix, targets *mat.Dense) float64 {
	var tmp mat.Dense
	var sum float64
	tmp.Apply(func(i, j int, v float64) float64 {
		sum += math.Abs(v - targets.At(i, j))
		return 0
	}, predict)
	rows, _ := predict.Dims()
	return sum / float64(rows)
}

func (*MAE) Grad(predict mat.Matrix, targets *mat.Dense) *mat.Dense {
	rows, _ := predict.Dims()
	var grad mat.Dense
	grad.Apply(func(i, j int, v float64) float64 {
		n := v - targets.At(i, j)
		if n > 0 {
			return 1 / float64(rows)
		} else if n < 0 {
			return -1 / float64(rows)
		}
		return 0
	}, predict)
	return &grad
}

func (loss *MAE) Save() *pb.Loss {
	return &pb.Loss{Name: "mae"}
}
