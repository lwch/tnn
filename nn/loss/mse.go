package loss

import (
	"github.com/lwch/tnn/internal/pb"
	"gonum.org/v1/gonum/mat"
)

type MSE struct{}

func NewMSE() *MSE {
	return &MSE{}
}

func (*MSE) Name() string {
	return "mse"
}

func (*MSE) Loss(predict, targets mat.Matrix) float64 {
	var tmp mat.Dense
	tmp.Sub(predict, targets)
	tmp.MulElem(&tmp, &tmp)
	rows, _ := predict.Dims()
	return 0.5 * mat.Sum(&tmp) / float64(rows)
}

func (*MSE) Grad(predict, targets mat.Matrix) mat.Matrix {
	var grad mat.Dense
	grad.Sub(predict, targets)
	rows, _ := predict.Dims()
	grad.Scale(1/float64(rows), &grad)
	return &grad
}

func (loss *MSE) Save() *pb.Loss {
	return &pb.Loss{Name: "mse"}
}
