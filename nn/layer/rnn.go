package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type Rnn struct {
	base
	featureSize, steps int
	hidden             int
	// params
	w *tensor.Tensor
	b *tensor.Tensor
}

func NewRnn(featureSize, steps, hidden int, opts ...LayerCreateOption) *Rnn {
	var layer Rnn
	layer.new("rnn", opts...)
	layer.featureSize = featureSize
	layer.steps = steps
	layer.hidden = hidden
	layer.w = layer.initW(int64(featureSize+layer.hidden), int64(hidden))
	layer.b = layer.initB(int64(hidden))
	return &layer
}

func LoadRnn(device consts.DeviceType, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Rnn
	layer.new("rnn", WithDevice(device))
	layer.name = name
	layer.featureSize = int(args["feature_size"])
	layer.steps = int(args["steps"])
	layer.hidden = int(args["hidden"])
	layer.w = layer.loadParam(params["w"])
	layer.b = layer.loadParam(params["b"])
	return &layer
}

func copyState(s *tensor.Tensor) *tensor.Tensor {
	return tensor.FromFloat32(nil, s.Float32Value(), tensor.WithShapes(s.Shapes()...))
}

func (layer *Rnn) Forward(x, h *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	inputShape := x.Shapes()
	if h == nil {
		h = tensor.Zeros(x.Storage(), consts.KFloat, tensor.WithShapes(inputShape[0], int64(layer.hidden)))
	}
	x = x.Transpose(1, 0) // (steps, batch, feature)
	var result *tensor.Tensor
	for step := 0; step < layer.steps; step++ {
		t := x.NArrow(0, int64(step), 1).
			Reshape(int64(inputShape[0]), int64(layer.featureSize)) // (batch, feature)
		z := tensor.HStack(t, h)           // (batch, feature+hidden)
		z = z.MatMul(layer.w).Add(layer.b) // (batch, hidden)
		h = z.Tanh()                       // (batch, hidden)
		if result == nil {
			result = h
		} else {
			result = tensor.VStack(result, h)
		}
	}
	return result.Reshape(inputShape[0], inputShape[1], int64(layer.hidden)), copyState(h)
}

func (layer *Rnn) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": layer.w,
		"b": layer.b,
	}
}

func (layer *Rnn) Args() map[string]float32 {
	return map[string]float32{
		"feature_size": float32(layer.featureSize),
		"steps":        float32(layer.steps),
		"hidden":       float32(layer.hidden),
	}
}

func (layer *Rnn) Freeze() {
	layer.w.SetRequiresGrad(false)
	layer.b.SetRequiresGrad(false)
}

func (layer *Rnn) Unfreeze() {
	layer.w.SetRequiresGrad(true)
	layer.b.SetRequiresGrad(true)
}
