package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type Nor struct {
	*base
	eps *tensor.Tensor
}

func NewNor(device consts.DeviceType) *Nor {
	var layer Nor
	layer.base = new("nor", device)
	return &layer
}

func LoadNor(device consts.DeviceType, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Nor
	layer.base = new("nor", device)
	layer.name = name
	return &layer
}

func (layer *Nor) Forward(x *tensor.Tensor) *tensor.Tensor {
	if layer.eps == nil {
		layer.eps = tensor.FromFloat32(nil, []float32{1e-9}, tensor.WithShapes(1), tensor.WithDevice(layer.device))
	}
	mean := x.Mean(-1, true)
	v := x.Var(-1, false, true)
	return x.Sub(mean).Div(v.Add(layer.eps).Sqrt())
}
