package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type Dropout struct {
	base
	keep float64
}

func NewDropout(keep float64) *Dropout {
	var layer Dropout
	layer.new("dropout")
	layer.keep = keep
	return &layer
}

func LoadDropout(_ consts.DeviceType, name string, _ map[string]*pb.Dense, args map[string]float64) Layer {
	var layer Dropout
	layer.new("dropout")
	layer.name = name
	layer.keep = args["keep"]
	return &layer
}

func (layer *Dropout) Forward(x *tensor.Tensor, train bool) *tensor.Tensor {
	return x.Dropout(layer.keep, train)
}

func (layer *Dropout) Args() map[string]float64 {
	return map[string]float64{
		"keep": layer.keep,
	}
}
