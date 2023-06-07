package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

type Rnn struct {
	*base
	featureSize, steps int
	hidden             int
	// params
	w *ts.Tensor
	b *ts.Tensor
}

func NewRnn(featureSize, steps, hidden int) *Rnn {
	var layer Rnn
	layer.base = new("rnn")
	layer.featureSize = featureSize
	layer.steps = steps
	layer.hidden = hidden
	return &layer
}

func LoadRnn(vs *nn.Path, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Rnn
	layer.base = new("rnn")
	layer.name = name
	layer.featureSize = int(args["feature_size"])
	layer.steps = int(args["steps"])
	layer.hidden = int(args["hidden"])
	layer.w = loadParam(vs, params["w"], "w")
	layer.b = loadParam(vs, params["b"], "b")
	return &layer
}

func (layer *Rnn) Forward(vs *nn.Path, x, h *ts.Tensor) (*ts.Tensor, *ts.Tensor) {
	inputShape := x.MustSize()
	if layer.w == nil {
		layer.w = initW(vs, "w", int64(layer.featureSize+layer.hidden), int64(layer.hidden))
	}
	if layer.b == nil {
		layer.b = initB(vs, "b", inputShape[0], int64(layer.hidden))
	}
	if h == nil {
		h = ts.MustZeros([]int64{inputShape[0], int64(layer.hidden)}, gotch.Float, vs.Device())
	}
	x = x.MustTranspose(1, 0, true) // (steps, batch, feature)
	var result *ts.Tensor
	for step := 0; step < layer.steps; step++ {
		t := x.MustNarrow(0, int64(step), 1, false).
			MustReshape([]int64{int64(inputShape[0]), int64(layer.featureSize)}, true) // (batch, feature)
		z := ts.MustHstack([]ts.Tensor{*h, *t}) // (batch, feature+hidden)
		z = z.MustMm(layer.w, true)             // (batch, hidden)
		z = z.MustAdd(layer.b, true)            // (batch, hidden)
		h = z.MustTanh(true)                    // (batch, hidden)
		if result == nil {
			result = h
		} else {
			result = ts.MustVstack([]ts.Tensor{*result, *h})
		}
	}
	return result.MustReshape([]int64{inputShape[0], inputShape[1], int64(layer.hidden)}, true), h
}

func (layer *Rnn) Params() map[string]*ts.Tensor {
	return map[string]*ts.Tensor{
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
