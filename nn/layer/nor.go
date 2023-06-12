package layer

import (
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type Nor struct {
	*base
}

func NewNor(output int) *Nor {
	var layer Nor
	layer.base = new("nor")
	return &layer
}

func LoadNor(name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Nor
	layer.base = new("nor")
	layer.name = name
	return &layer
}

func (layer *Nor) Forward(x *tensor.Tensor) *tensor.Tensor {
	// TODO
	return x
}
