package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

type ConvTranspose2D struct {
	base
	inC, outC     int
	kernel        [2]int
	stride        [2]int
	padding       [2]int
	outputPadding [2]int
	dilation      int
	groups        int
	// params
	w *tensor.Tensor
}

func NewConvTranspose2D(name string, inC, outC int, kernel1, kernel2 int, opts ...LayerCreateOption) *ConvTranspose2D {
	var layer ConvTranspose2D
	layer.new("convtranspose2d", name, opts...)
	layer.inC = inC
	layer.outC = outC
	layer.kernel = [2]int{kernel1, kernel2}
	layer.stride = [2]int{1, 1}
	layer.padding = [2]int{0, 0}
	layer.outputPadding = [2]int{0, 0}
	layer.dilation = 1
	layer.groups = 1
	layer.w = layer.initW(int64(inC), int64(outC), int64(kernel1), int64(kernel2))
	return &layer
}

func (layer *ConvTranspose2D) SetStride(stride1, stride2 int) {
	layer.stride = [2]int{stride1, stride2}
}

func (layer *ConvTranspose2D) SetPadding(padding1, padding2 int) {
	layer.padding = [2]int{padding1, padding2}
}

func (layer *ConvTranspose2D) SetOutputPadding(padding1, padding2 int) {
	layer.outputPadding = [2]int{padding1, padding2}
}

func (layer *ConvTranspose2D) SetDilation(dilation int) {
	layer.dilation = dilation
}

func (layer *ConvTranspose2D) SetGroups(groups int) {
	layer.groups = groups
}

func LoadConvTranspose2D(name string, params []*tensor.Tensor, args map[string]float32) Layer {
	var layer ConvTranspose2D
	layer.new("convtranspose2d", name)
	layer.inC = int(args["inC"])
	layer.outC = int(args["outC"])
	layer.kernel = [2]int{int(args["kernel1"]), int(args["kernel2"])}
	layer.stride = [2]int{int(args["stride1"]), int(args["stride2"])}
	layer.padding = [2]int{int(args["padding1"]), int(args["padding2"])}
	layer.outputPadding = [2]int{int(args["output_padding1"]), int(args["output_padding2"])}
	layer.dilation = int(args["dilation"])
	layer.groups = int(args["groups"])
	layer.w = params[0]
	return &layer
}

func (layer *ConvTranspose2D) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.ConvTranspose2D(layer.w, nil,
		tensor.ConvTranspose2DStride(layer.stride[0], layer.stride[1]),
		tensor.ConvTranspose2DPadding(layer.padding[0], layer.padding[1]),
		tensor.ConvTranspose2DOutputPadding(layer.outputPadding[0], layer.outputPadding[1]),
		tensor.ConvTranspose2DDilation(layer.dilation),
		tensor.ConvTranspose2DGroups(layer.groups))
}

func (layer *ConvTranspose2D) Params() []*tensor.Tensor {
	return []*tensor.Tensor{
		layer.w,
	}
}

func (layer *ConvTranspose2D) Args() map[string]float32 {
	return map[string]float32{
		"inC":             float32(layer.inC),
		"outC":            float32(layer.outC),
		"kernel1":         float32(layer.kernel[0]),
		"kernel2":         float32(layer.kernel[1]),
		"stride1":         float32(layer.stride[0]),
		"stride2":         float32(layer.stride[1]),
		"padding1":        float32(layer.padding[0]),
		"padding2":        float32(layer.padding[1]),
		"output_padding1": float32(layer.outputPadding[0]),
		"output_padding2": float32(layer.outputPadding[1]),
		"dilation":        float32(layer.dilation),
		"groups":          float32(layer.groups),
	}
}

func (layer *ConvTranspose2D) Freeze() {
	layer.w.SetRequiresGrad(false)
}

func (layer *ConvTranspose2D) Unfreeze() {
	layer.w.SetRequiresGrad(true)
}

func (layer *ConvTranspose2D) ToScalarType(t consts.ScalarType) {
	layer.w = layer.w.ToScalarType(t)
}

func (layer *ConvTranspose2D) Reset() {
	layer.w = layer.initW(layer.w.Shapes()...)
}

func (layer *ConvTranspose2D) Clone() Layer {
	return &ConvTranspose2D{
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

func (layer *ConvTranspose2D) Update(params []*tensor.Tensor) {
	layer.w = params[0]
}
