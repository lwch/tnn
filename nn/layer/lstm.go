package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
	"gorgonia.org/gorgonia"
)

type Lstm struct {
	*base
	featureSize, steps int
	hidden             int
	Wi, Bi             *ts.Tensor
	Wf, Bf             *ts.Tensor
	Wg, Bg             *ts.Tensor
	Wo, Bo             *ts.Tensor
}

func NewLstm(featureSize, steps, hidden int) *Lstm {
	var layer Lstm
	layer.base = new("lstm")
	layer.featureSize = featureSize
	layer.steps = steps
	layer.hidden = hidden
	return &layer
}

func LoadLstm(g *gorgonia.ExprGraph, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Lstm
	layer.base = new("lstm")
	layer.name = name
	layer.featureSize = int(args["feature_size"])
	layer.steps = int(args["steps"])
	layer.hidden = int(args["hidden"])
	layer.Wi = loadParam(params["Wi"])
	layer.Wf = loadParam(params["Wf"])
	layer.Wg = loadParam(params["Wg"])
	layer.Wo = loadParam(params["Wo"])
	layer.Bi = loadParam(params["Bi"])
	layer.Bf = loadParam(params["Bf"])
	layer.Bg = loadParam(params["Bg"])
	layer.Bo = loadParam(params["Bo"])
	return &layer
}

func (layer *Lstm) Forward(vs *nn.Path, x, h, c *ts.Tensor) (*ts.Tensor, *ts.Tensor, *ts.Tensor, error) {
	inputShape := x.MustSize()
	if layer.Wi == nil {
		layer.Wi = initW(vs, "Wi", int64(layer.featureSize+layer.hidden), int64(layer.hidden))
	}
	if layer.Wf == nil {
		layer.Wf = initW(vs, "Wf", int64(layer.featureSize+layer.hidden), int64(layer.hidden))
	}
	if layer.Wg == nil {
		layer.Wg = initW(vs, "Wg", int64(layer.featureSize+layer.hidden), int64(layer.hidden))
	}
	if layer.Wo == nil {
		layer.Wo = initW(vs, "Wo", int64(layer.featureSize+layer.hidden), int64(layer.hidden))
	}
	if layer.Bi == nil {
		layer.Bi = initB(vs, "Bi", inputShape[0], int64(layer.hidden))
	}
	if layer.Bf == nil {
		layer.Bf = initB(vs, "Bf", inputShape[0], int64(layer.hidden))
	}
	if layer.Bg == nil {
		layer.Bg = initB(vs, "Bg", inputShape[0], int64(layer.hidden))
	}
	if layer.Bo == nil {
		layer.Bo = initB(vs, "Bo", inputShape[0], int64(layer.hidden))
	}
	if h == nil {
		h = vs.MustZeros("h", []int64{int64(inputShape[0]), int64(layer.hidden)})
	}
	if c == nil {
		c = vs.MustZeros("c", []int64{int64(inputShape[0]), int64(layer.hidden)})
	}
	x = x.MustTranspose(1, 0, false) // (steps, batch, feature)
	var result *ts.Tensor
	for step := 0; step < layer.steps; step++ {
		t := x.MustNarrow(0, int64(step), 1, false).
			MustReshape([]int64{int64(inputShape[0]), int64(layer.featureSize)}, false) // (batch, feature)
		z := ts.MustHstack([]ts.Tensor{*h, *t})                 // (batch, feature+hidden)
		i := z.MustMm(layer.Wi, false).MustAdd(layer.Bi, false) // (batch, hidden)
		i = i.MustSigmoid(false)                                // (batch, hidden)
		f := z.MustMm(layer.Wf, false).MustAdd(layer.Bf, false) // (batch, hidden)
		f = f.MustSigmoid(false)                                // (batch, hidden)
		o := z.MustMm(layer.Wo, false).MustAdd(layer.Bo, false) // (batch, hidden)
		o = o.MustSigmoid(false)                                // (batch, hidden)
		g := z.MustMm(layer.Wg, false).MustAdd(layer.Bg, false) // (batch, hidden)
		g = g.MustTanh(false)                                   // (batch, hidden)
		a := f.MustMul(c, false)                                // (batch, hidden)
		b := i.MustMul(g, false)                                // (batch, hidden)
		c = a.MustAdd(b, false)                                 // (batch, hidden)
		ct := c.MustTanh(false)                                 // (batch, hidden)
		h = o.MustMul(ct, false)                                // (batch, hidden)
		if result == nil {
			result = h
		} else {
			result = ts.MustVstack([]ts.Tensor{*result, *h})
		}
	}
	return result.MustReshape([]int64{inputShape[0], inputShape[1], int64(layer.hidden)}, true), h, c, nil
}

func (layer *Lstm) Params() map[string]*ts.Tensor {
	return map[string]*ts.Tensor{
		"Wi": layer.Wi,
		"Wf": layer.Wf,
		"Wg": layer.Wg,
		"Wo": layer.Wo,
		"Bi": layer.Bi,
		"Bf": layer.Bf,
		"Bg": layer.Bg,
		"Bo": layer.Bo,
	}
}

func (layer *Lstm) Args() map[string]float32 {
	return map[string]float32{
		"feature_size": float32(layer.featureSize),
		"steps":        float32(layer.steps),
		"hidden":       float32(layer.hidden),
	}
}
