package loss

import "gorgonia.org/gorgonia"

type MSE struct {
	*base
}

func NewMSE() Loss {
	var loss MSE
	loss.base = new("mse")
	return &loss
}

func (mse *MSE) Loss(y, pred *gorgonia.Node) *gorgonia.Node {
	diff := gorgonia.Must(gorgonia.Sub(y, pred))
	sqDiff := gorgonia.Must(gorgonia.Square(diff))
	return gorgonia.Must(gorgonia.Mean(sqDiff))
}
