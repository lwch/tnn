package loss

import (
	"github.com/lwch/tnn/internal/pb"
	"gorgonia.org/gorgonia"
)

type Loss interface {
	Loss(y, pred *gorgonia.Node) *gorgonia.Node
	Save() *pb.Loss
}

func Load(loss *pb.Loss) Loss {
	switch loss.Name {
	case "mse":
		return NewMSE()
	// case "mae":
	// 	return NewMAE()
	// case "huber":
	// 	return NewHuber(loss.GetParams()["delta"])
	// case "softmax":
	// 	return NewSoftmax()
	// case "sigmoid":
	// 	return NewSigmoid()
	default:
		panic("unsupported " + loss.Name + " loss function")
	}
}

type base struct {
	name string
}

func new(name string) *base {
	return &base{
		name: name,
	}
}

func (b *base) Name() string {
	return b.name
}

func (b *base) Save() *pb.Loss {
	return &pb.Loss{
		Name: b.name,
	}
}
