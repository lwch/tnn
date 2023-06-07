package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/sugarme/gotch/ts"
)

type Flatten struct {
	*base
}

func NewFlatten() *Flatten {
	var layer Flatten
	layer.base = new("flatten")
	return &layer
}

func LoadFlatten(name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Flatten
	layer.base = new("flatten")
	layer.name = name
	return &layer
}

func (layer *Flatten) Forward(x *ts.Tensor) *ts.Tensor {
	shape := x.MustSize()
	cols := int64(1)
	for _, v := range shape[1:] {
		cols *= v
	}
	return x.MustReshape([]int64{shape[0], cols}, true)
}
