package activation

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/layer"
	"github.com/sugarme/gotch/nn"
	"gorgonia.org/gorgonia"
)

type Sigmoid struct {
	*base
}

func NewSigmoid() *Sigmoid {
	var layer Sigmoid
	layer.base = new("sigmoid")
	return &layer
}

func LoadSigmoid(_ *nn.Path, name string, _ map[string]*pb.Dense, _ map[string]float32) layer.Layer {
	var layer Sigmoid
	layer.base = new("sigmoid")
	layer.name = name
	return &layer
}

func (layer *Sigmoid) Forward(x *gorgonia.Node) *gorgonia.Node {
	return gorgonia.Must(gorgonia.Sigmoid(x))
}
