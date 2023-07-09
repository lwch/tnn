package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type LayerNorm struct {
	*base
	eps *tensor.Tensor
	// params
	w *tensor.Tensor
}

func NewLayerNorm(device consts.DeviceType) *LayerNorm {
	var layer LayerNorm
	layer.base = new("layer_norm", device)
	return &layer
}

func LoadLayerNorm(device consts.DeviceType, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer LayerNorm
	layer.base = new("layer_norm", device)
	layer.name = name
	layer.w = layer.loadParam(params["w"])
	return &layer
}

func (layer *LayerNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	if layer.eps == nil {
		layer.eps = tensor.FromFloat32(nil, []float32{1e-9}, tensor.WithShapes(1), tensor.WithDevice(layer.device))
	}
	if layer.w == nil {
		layer.w = layer.initW(lastDim(x))
	}
	mean := x.Mean(-1, true)
	v := x.Var(-1, false, true)
	return layer.w.Mul(x.Sub(mean)).Div(v.Add(layer.eps).Sqrt())
}

func (layer *LayerNorm) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": layer.w,
	}
}

func (layer *LayerNorm) Freeze() {
	if layer.w != nil {
		layer.w.SetRequiresGrad(false)
	}
}

func (layer *LayerNorm) Unfreeze() {
	if layer.w != nil {
		layer.w.SetRequiresGrad(true)
	}
}
