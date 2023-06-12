package layer

import (
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type Dropout struct {
	*base
	keep float64
}

func NewDropout(keep float64) *Dropout {
	var layer Dropout
	layer.base = new("dropout")
	layer.keep = keep
	return &layer
}

func LoadDropout(name string, _ map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Dropout
	layer.base = new("dropout")
	layer.name = name
	layer.keep = float64(args["keep"])
	return &layer
}

func (layer *Dropout) Forward(x *tensor.Tensor, train bool) *tensor.Tensor {
	return x.Dropout(layer.keep, train)
}

func (layer *Dropout) Args() map[string]float32 {
	return map[string]float32{
		"keep": float32(layer.keep),
	}
}
