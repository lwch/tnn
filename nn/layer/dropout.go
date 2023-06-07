package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
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

func LoadDropout(_ *nn.Path, name string, _ map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Dropout
	layer.base = new("dropout")
	layer.name = name
	layer.keep = float64(args["keep"])
	return &layer
}

func (layer *Dropout) Forward(x *ts.Tensor, train bool) *ts.Tensor {
	x.MustDropout_(layer.keep, train)
	return x
}

func (layer *Dropout) Args() map[string]float32 {
	return map[string]float32{
		"keep": float32(layer.keep),
	}
}
