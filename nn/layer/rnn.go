package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type Rnn struct {
	*base
	featureSize, steps int
	hidden             int
	// params
	w *tensor.Tensor
	b *tensor.Tensor
}

func NewRnn(featureSize, steps, hidden int) *Rnn {
	var layer Rnn
	layer.base = new("rnn")
	layer.featureSize = featureSize
	layer.steps = steps
	layer.hidden = hidden
	return &layer
}

func LoadRnn(name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Rnn
	layer.base = new("rnn")
	layer.name = name
	layer.featureSize = int(args["feature_size"])
	layer.steps = int(args["steps"])
	layer.hidden = int(args["hidden"])
	layer.w = loadParam(params["w"])
	layer.b = loadParam(params["b"])
	return &layer
}

func (layer *Rnn) Forward(x, h *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	inputShape := x.Shapes()
	if layer.w == nil {
		layer.w = initW(int64(layer.featureSize+layer.hidden), int64(layer.hidden))
	}
	if layer.b == nil {
		layer.b = initB(int64(layer.hidden))
	}
	if h == nil {
		h = tensor.Zeros(nil, consts.KFloat, inputShape[0], int64(layer.hidden))
	}
	// TODO
	return x, h
	// x = x.MustTranspose(1, 0, true) // (steps, batch, feature)
	// var result *ts.Tensor
	// for step := 0; step < layer.steps; step++ {
	// 	t := x.MustNarrow(0, int64(step), 1, false).
	// 		MustReshape([]int64{int64(inputShape[0]), int64(layer.featureSize)}, true) // (batch, feature)
	// 	z := ts.MustHstack([]ts.Tensor{*h, *t}) // (batch, feature+hidden)
	// 	z = z.MustMm(layer.w, true)             // (batch, hidden)
	// 	z = z.MustAdd(layer.b, true)            // (batch, hidden)
	// 	h = z.MustTanh(true)                    // (batch, hidden)
	// 	if result == nil {
	// 		result = h
	// 	} else {
	// 		result = ts.MustVstack([]ts.Tensor{*result, *h})
	// 	}
	// }
	// return result.MustReshape([]int64{inputShape[0], inputShape[1], int64(layer.hidden)}, true), h
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
