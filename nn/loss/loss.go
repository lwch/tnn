package loss

import (
	"fmt"

	"github.com/lwch/tnn/nn/pb"
	"gonum.org/v1/gonum/mat"
)

type Loss interface {
	Name() string
	Loss(predict, targets mat.Matrix) float64
	Grad(predict, targets mat.Matrix) mat.Matrix
	Save() *pb.Loss
}

func Load(loss *pb.Loss) Loss {
	switch loss.Name {
	case "mse":
		return NewMSE()
	case "mae":
		return NewMAE()
	case "huber":
		return NewHuber(loss.GetParams()["delta"])
	case "softmax":
		return NewSoftmax(loss.GetParams()["t"])
	default:
		panic("unsupported " + loss.Name + " loss function")
	}
}

func Print(loss Loss) {
	fmt.Println("Loss Func:", loss.Name())
}
