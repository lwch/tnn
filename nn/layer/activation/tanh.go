package activation

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
)

type Tanh struct {
	*base
}

func NewTanh() *Tanh {
	var layer Tanh
	layer.base = new("tanh")
	return &layer
}

func LoadTanh(_ consts.DeviceType, name string, _ map[string]*tensor.Tensor, _ map[string]float32) layer.Layer {
	var layer Tanh
	layer.base = new("tanh")
	layer.name = name
	return &layer
}

func (layer *Tanh) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.Tanh()
}
