package activation

import (
	"github.com/lwch/tnn/nn/layer"
	"gorgonia.org/gorgonia"
)

type ReLU struct {
}

func NewReLU() layer.Layer {
	return &ReLU{}
}

func (layer *ReLU) Forward(x *gorgonia.Node) *gorgonia.Node {
	return gorgonia.Must(gorgonia.Rectify(x))
}

func (layer *ReLU) Params() gorgonia.Nodes {
	return nil
}
