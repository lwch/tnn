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

func (*MAE) Loss(predict, targets *mat.Dense) float64 {
	var tmp mat.Dense
	var sum float64
	tmp.Apply(func(i, j int, v float64) float64 {
		sum += math.Abs(v - targets.At(i, j))
		return 0
	}, predict)
	return sum / float64(predict.RawMatrix().Rows)
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
