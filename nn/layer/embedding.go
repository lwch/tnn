package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type Embedding struct {
	base
	num     int
	dim     int
	padding int64
	// params
	w *tensor.Tensor
}

func NewEmbedding(num, dim int, opts ...LayerCreateOption) *Embedding {
	var layer Embedding
	layer.new("embedding", opts...)
	layer.num = num
	layer.dim = dim
	layer.padding = -1
	return &layer
}

func LoadEmbedding(device consts.DeviceType, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Embedding
	layer.new("embedding", WithDevice(device))
	layer.name = name
	layer.num = int(args["num"])
	layer.dim = int(args["dim"])
	layer.padding = int64(args["padding"])
	layer.w = layer.loadParam(params["w"])
	return &layer
}

func (layer *Embedding) SetPaddingIdx(n int64) {
	layer.padding = n
}

func (layer *Embedding) Forward(x *tensor.Tensor) *tensor.Tensor {
	if layer.w == nil {
		layer.w = layer.initW(int64(layer.num), int64(layer.dim))
	}
	return tensor.Embedding(x, layer.w, layer.padding)
}

func (layer *Embedding) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": layer.w,
	}
}

func (layer *Embedding) Args() map[string]float32 {
	return map[string]float32{
		"num":     float32(layer.num),
		"dim":     float32(layer.dim),
		"padding": float32(layer.padding),
	}
}

func (layer *Embedding) Freeze() {
	if layer.w != nil {
		layer.w.SetRequiresGrad(false)
	}
}

func (layer *Embedding) Unfreeze() {
	if layer.w != nil {
		layer.w.SetRequiresGrad(true)
	}
}
