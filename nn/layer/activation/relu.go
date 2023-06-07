package activation

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/layer"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

type ReLU struct {
	*base
}

func NewReLU() *ReLU {
	var layer ReLU
	layer.base = new("relu")
	return &layer
}

func LoadRelu(_ *nn.Path, name string, _ map[string]*pb.Dense, _ map[string]float32) layer.Layer {
	var layer ReLU
	layer.base = new("relu")
	layer.name = name
	return &layer
}

func (layer *ReLU) Forward(x *ts.Tensor) *ts.Tensor {
	return x.MustRelu(true)
}
