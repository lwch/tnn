package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

type Conv1D struct {
	base
	inC, outC int
	kernel    int
	stride    int
	padding   int
	dilation  int
	groups    int
	// params
	w *tensor.Tensor
}

func NewConv1D(name string, inC, outC, kernel int, opts ...LayerCreateOption) *Conv1D {
	var layer Conv1D
	layer.new("conv1d", name, opts...)
	layer.inC = inC
	layer.outC = outC
	layer.kernel = kernel
	layer.stride = 1
	layer.padding = 0
	layer.dilation = 1
	layer.groups = 1
	layer.w = layer.initW(int64(outC), int64(inC), int64(kernel))
	return &layer
}

func (layer *Conv1D) SetStride(stride int) {
	layer.stride = stride
}

func (layer *Conv1D) SetPadding(padding int) {
	layer.padding = padding
}

func (layer *Conv1D) SetDilation(dilation int) {
	layer.dilation = dilation
}

func (layer *Conv1D) SetGroups(groups int) {
	layer.groups = groups
}

func LoadConv1D(name string, params []*tensor.Tensor, args map[string]float32) Layer {
	var layer Conv1D
	layer.new("conv1d", name)
	layer.inC = int(args["inC"])
	layer.outC = int(args["outC"])
	layer.kernel = int(args["kernel"])
	layer.stride = int(args["stride"])
	layer.padding = int(args["padding"])
	layer.dilation = int(args["dilation"])
	layer.groups = int(args["groups"])
	layer.w = params[0]
	return &layer
}

func (layer *Conv1D) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.Conv1D(layer.w, nil,
		tensor.Conv1DStride(layer.stride),
		tensor.Conv1DPadding(layer.padding),
		tensor.Conv1DDilation(layer.dilation),
		tensor.Conv1DGroups(layer.groups))
}

func (layer *Conv1D) Params() []*tensor.Tensor {
	return []*tensor.Tensor{
		layer.w,
	}
}

func (layer *Conv1D) Args() map[string]float32 {
	return map[string]float32{
		"inC":      float32(layer.inC),
		"outC":     float32(layer.outC),
		"kernel":   float32(layer.kernel),
		"stride":   float32(layer.stride),
		"padding":  float32(layer.padding),
		"dilation": float32(layer.dilation),
		"groups":   float32(layer.groups),
	}
}

func (layer *Conv1D) Freeze() {
	layer.w.SetRequiresGrad(false)
}

func (layer *Conv1D) Unfreeze() {
	layer.w.SetRequiresGrad(true)
}

func (layer *Conv1D) ToScalarType(t consts.ScalarType) {
	layer.w = layer.w.ToScalarType(t)
}

func (layer *Conv1D) Reset() {
	layer.w = layer.initW(layer.w.Shapes()...)
}

func (layer *Conv1D) Clone() Layer {
	return &Conv1D{
		base:     layer.base.clone(),
		inC:      layer.inC,
		outC:     layer.outC,
		kernel:   layer.kernel,
		stride:   layer.stride,
		padding:  layer.padding,
		dilation: layer.dilation,
		groups:   layer.groups,
		w:        layer.w.Clone(),
	}
}

func (layer *Conv1D) Update(params []*tensor.Tensor) {
	layer.w = params[0]
}
