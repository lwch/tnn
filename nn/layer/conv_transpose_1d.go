package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

type ConvTranspose1D struct {
	base
	inC, outC     int
	kernel        int
	stride        int
	padding       int
	outputPadding int
	dilation      int
	groups        int
	// params
	w *tensor.Tensor
}

func NewConvTranspose1D(name string, inC, outC, kernel int, opts ...LayerCreateOption) *ConvTranspose1D {
	var layer ConvTranspose1D
	layer.new("convtranspose1d", name, opts...)
	layer.inC = inC
	layer.outC = outC
	layer.kernel = kernel
	layer.stride = 1
	layer.padding = 0
	layer.outputPadding = 0
	layer.dilation = 1
	layer.groups = 1
	layer.w = layer.initW(int64(inC), int64(outC), int64(kernel))
	return &layer
}

func (layer *ConvTranspose1D) SetStride(stride int) {
	layer.stride = stride
}

func (layer *ConvTranspose1D) SetPadding(padding int) {
	layer.padding = padding
}

func (layer *ConvTranspose1D) SetOutputPadding(padding int) {
	layer.outputPadding = padding
}

func (layer *ConvTranspose1D) SetDilation(dilation int) {
	layer.dilation = dilation
}

func (layer *ConvTranspose1D) SetGroups(groups int) {
	layer.groups = groups
}

func LoadConvTranspose1D(name string, params []*tensor.Tensor, args map[string]float32) Layer {
	var layer ConvTranspose1D
	layer.new("convtranspose1d", name)
	layer.inC = int(args["inC"])
	layer.outC = int(args["outC"])
	layer.kernel = int(args["kernel"])
	layer.stride = int(args["stride"])
	layer.padding = int(args["padding"])
	layer.outputPadding = int(args["output_padding"])
	layer.dilation = int(args["dilation"])
	layer.groups = int(args["groups"])
	layer.w = params[0]
	return &layer
}

func (layer *ConvTranspose1D) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.ConvTranspose1D(layer.w, nil,
		tensor.ConvTranspose1DStride(layer.stride),
		tensor.ConvTranspose1DPadding(layer.padding),
		tensor.ConvTranspose1DOutputPadding(layer.outputPadding),
		tensor.ConvTranspose1DDilation(layer.dilation),
		tensor.ConvTranspose1DGroups(layer.groups))
}

func (layer *ConvTranspose1D) Params() []*tensor.Tensor {
	return []*tensor.Tensor{
		layer.w,
	}
}

func (layer *ConvTranspose1D) Args() map[string]float32 {
	return map[string]float32{
		"inC":            float32(layer.inC),
		"outC":           float32(layer.outC),
		"kernel":         float32(layer.kernel),
		"stride":         float32(layer.stride),
		"padding":        float32(layer.padding),
		"output_padding": float32(layer.outputPadding),
		"dilation":       float32(layer.dilation),
		"groups":         float32(layer.groups),
	}
}

func (layer *ConvTranspose1D) Freeze() {
	layer.w.SetRequiresGrad(false)
}

func (layer *ConvTranspose1D) Unfreeze() {
	layer.w.SetRequiresGrad(true)
}

func (layer *ConvTranspose1D) ToScalarType(t consts.ScalarType) {
	layer.w = layer.w.ToScalarType(t)
}

func (layer *ConvTranspose1D) Reset() {
	layer.w = layer.initW(layer.w.Shapes()...)
}

func (layer *ConvTranspose1D) Clone() Layer {
	return &ConvTranspose1D{
		base:          layer.base.clone(),
		inC:           layer.inC,
		outC:          layer.outC,
		kernel:        layer.kernel,
		stride:        layer.stride,
		padding:       layer.padding,
		outputPadding: layer.outputPadding,
		dilation:      layer.dilation,
		groups:        layer.groups,
		w:             layer.w.Clone(),
	}
}
