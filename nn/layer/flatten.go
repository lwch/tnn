package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

type Flatten struct {
	base
}

func NewFlatten(name string) *Flatten {
	var layer Flatten
	layer.new("flatten", name)
	return &layer
}

func LoadFlatten(name string, _ []*tensor.Tensor, _ map[string]float32) Layer {
	var layer Flatten
	layer.new("flatten", name)
	return &layer
}

func (layer *Flatten) Forward(x *tensor.Tensor) *tensor.Tensor {
	shape := x.Shapes()
	cols := int64(1)
	for _, v := range shape[1:] {
		cols *= v
	}
	return x.Reshape(shape[0], cols)
}

func (layer *Flatten) ToScalarType(t consts.ScalarType) {
}

func (layer *Flatten) Reset() {
}

func (layer *Flatten) Clone() Layer {
	return &Flatten{
		base: layer.base.clone(),
	}
}

func (layer *Flatten) Update(_ []*tensor.Tensor) {
}
