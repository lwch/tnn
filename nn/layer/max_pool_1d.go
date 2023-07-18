package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type MaxPool1D struct {
	base
	kernel   int
	stride   int
	padding  int
	dilation int
	ceil     bool
}

func NewMaxPool1D(kernel int, opts ...LayerCreateOption) *MaxPool1D {
	var layer MaxPool1D
	layer.new("maxpool1d", opts...)
	layer.kernel = kernel
	layer.stride = -1
	layer.padding = 0
	layer.dilation = 1
	layer.ceil = false
	return &layer
}

func (layer *MaxPool1D) SetStride(stride int) {
	layer.stride = stride
}

func (layer *MaxPool1D) SetPadding(padding int) {
	layer.padding = padding
}

func (layer *MaxPool1D) SetDilation(dilation int) {
	layer.dilation = dilation
}

func (layer *MaxPool1D) SetCeil(ceil bool) {
	layer.ceil = ceil
}

func LoadMaxPool1D(device consts.DeviceType, name string, _ map[string]*pb.Dense, args map[string]float32) Layer {
	var layer MaxPool1D
	layer.new("maxpool1d", WithDevice(device))
	layer.name = name
	layer.kernel = int(args["kernel"])
	layer.stride = int(args["stride"])
	layer.padding = int(args["padding"])
	layer.dilation = int(args["dilation"])
	layer.ceil = args["ceil"] > 0
	return &layer
}

func (layer *MaxPool1D) Forward(x *tensor.Tensor) *tensor.Tensor {
	if layer.stride < 0 {
		layer.stride = layer.kernel
	}
	return x.MaxPool1D(layer.kernel,
		tensor.PoolStride(layer.stride),
		tensor.PoolPadding(layer.padding),
		tensor.PoolDilation(layer.dilation),
		tensor.PoolCeil(layer.ceil))
}

func (layer *MaxPool1D) Args() map[string]float32 {
	var ceil float32
	if layer.ceil {
		ceil = 1
	}
	return map[string]float32{
		"kernel":   float32(layer.kernel),
		"stride":   float32(layer.stride),
		"padding":  float32(layer.padding),
		"dilation": float32(layer.dilation),
		"ceil":     ceil,
	}
}
