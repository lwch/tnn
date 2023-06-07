package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

type Nor struct {
	*base
}

func NewNor(output int) *Nor {
	var layer Nor
	layer.base = new("nor")
	return &layer
}

func LoadNor(_ *nn.Path, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Nor
	layer.base = new("nor")
	layer.name = name
	return &layer
}

func (layer *Nor) Forward(x *ts.Tensor) *ts.Tensor {
	// TODO
	return x
}
