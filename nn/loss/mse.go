package loss

import "gorgonia.org/gorgonia"

type MSE struct{}

func NewMSE() Loss {
	return &MSE{}
}

func (mse *MSE) Loss(y, pred *gorgonia.Node) *gorgonia.Node {
	diff := gorgonia.Must(gorgonia.Sub(y, pred))
	sqDiff := gorgonia.Must(gorgonia.Square(diff))
	return gorgonia.Must(gorgonia.Mean(sqDiff))
}
