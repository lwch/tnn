package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type Conv1D struct {
	*base
	inC, outC int
	kernel    int
	stride    int
	padding   int
	dilation  int
	groups    int
	// params
	w *tensor.Tensor
	b *tensor.Tensor
}

func NewConv1D(inC, outC, kernel int, device consts.DeviceType) *Conv1D {
	var layer Conv1D
	layer.base = new("conv1d", device)
	layer.inC = inC
	layer.outC = outC
	layer.kernel = kernel
	layer.stride = 1
	layer.padding = 0
	layer.dilation = 1
	layer.groups = 1
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

func LoadConv1D(device consts.DeviceType, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Conv1D
	layer.base = new("conv1d", device)
	layer.name = name
	layer.inC = int(args["inC"])
	layer.outC = int(args["outC"])
	layer.kernel = int(args["kernel"])
	layer.stride = int(args["stride"])
	layer.padding = int(args["padding"])
	layer.dilation = int(args["dilation"])
	layer.groups = int(args["groups"])
	layer.w = layer.loadParam(params["w"])
	layer.b = layer.loadParam(params["b"])
	return &layer
}

func (layer *Conv1D) Forward(x *tensor.Tensor) *tensor.Tensor {
	if layer.w == nil {
		layer.w = layer.initW(int64(layer.outC), int64(layer.inC/layer.groups), int64(layer.kernel))
	}
	if layer.b == nil {
		layer.b = layer.initB(int64(layer.outC))
	}
	return x.Conv1D(layer.w, layer.b,
		tensor.ConvStride(layer.stride),
		tensor.ConvPadding(layer.padding),
		tensor.ConvDilation(layer.dilation),
		tensor.ConvGroups(layer.groups))
}

func (layer *Conv1D) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": layer.w,
		"b": layer.b,
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
	if layer.w != nil {
		layer.w.SetRequiresGrad(false)
	}
	if layer.b != nil {
		layer.b.SetRequiresGrad(false)
	}
}

func (layer *Conv1D) Unfreeze() {
	if layer.w != nil {
		layer.w.SetRequiresGrad(true)
	}
	if layer.b != nil {
		layer.b.SetRequiresGrad(true)
	}
}
