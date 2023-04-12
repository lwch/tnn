package loss

import (
	"fmt"
	"tnn/internal/nn/pb"

	"gonum.org/v1/gonum/mat"
)

type Loss interface {
	Name() string
	Loss(predict, targets *mat.Dense) float64
	Grad(predict, targets *mat.Dense) *mat.Dense
	Save() *pb.Loss
}

func Load(loss *pb.Loss) Loss {
	switch loss.Name {
	case "mse":
		return NewMSE()
	case "mae":
		return NewMAE()
	default:
		return nil
	}
}

func Print(loss Loss) {
	fmt.Println("Loss Func:", loss.Name())
}
