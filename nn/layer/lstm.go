package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type Lstm struct {
	*base
	featureSize, steps int
	hidden             int
	Wi, Bi             *tensor.Tensor
	Wf, Bf             *tensor.Tensor
	Wg, Bg             *tensor.Tensor
	Wo, Bo             *tensor.Tensor
}

func NewLstm(featureSize, steps, hidden int) *Lstm {
	var layer Lstm
	layer.base = new("lstm")
	layer.featureSize = featureSize
	layer.steps = steps
	layer.hidden = hidden
	return &layer
}

func LoadLstm(name string, params map[string]*pb.Dense, args map[string]float32) Layer {
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

func (layer *Lstm) Forward(x, h, c *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	inputShape := x.Shapes()
	if layer.Wi == nil {
		layer.Wi = initW(int64(layer.featureSize+layer.hidden), int64(layer.hidden))
	}
	if layer.Wf == nil {
		layer.Wf = initW(int64(layer.featureSize+layer.hidden), int64(layer.hidden))
	}
	if layer.Wg == nil {
		layer.Wg = initW(int64(layer.featureSize+layer.hidden), int64(layer.hidden))
	}
	if layer.Wo == nil {
		layer.Wo = initW(int64(layer.featureSize+layer.hidden), int64(layer.hidden))
	}
	if layer.Bi == nil {
		layer.Bi = initB(int64(layer.hidden))
	}
	if layer.Bf == nil {
		layer.Bf = initB(int64(layer.hidden))
	}
	if layer.Bg == nil {
		layer.Bg = initB(int64(layer.hidden))
	}
	if layer.Bo == nil {
		layer.Bo = initB(int64(layer.hidden))
	}
	if h == nil {
		h = tensor.Zeros(nil, consts.KFloat, int64(inputShape[0]), int64(layer.hidden))
	}
	if c == nil {
		c = tensor.Zeros(nil, consts.KFloat, int64(inputShape[0]), int64(layer.hidden))
	}
	x = x.Transpose(1, 0) // (steps, batch, feature)
	var result *tensor.Tensor
	for step := 0; step < layer.steps; step++ {
		t := x.NArrow(0, int64(step), 1).
			Reshape(int64(inputShape[0]), int64(layer.featureSize)) // (batch, feature)
		z := tensor.HStack(t, h)              // (batch, feature+hidden)
		i := z.MatMul(layer.Wi).Add(layer.Bi) // (batch, hidden)
		i = i.Sigmoid()                       // (batch, hidden)
		f := z.MatMul(layer.Wf).Add(layer.Bf) // (batch, hidden)
		f = f.Sigmoid()                       // (batch, hidden)
		o := z.MatMul(layer.Wo).Add(layer.Bo) // (batch, hidden)
		o = o.Sigmoid()                       // (batch, hidden)
		g := z.MatMul(layer.Wg).Add(layer.Bg) // (batch, hidden)
		g = g.Tanh()                          // (batch, hidden)
		a := f.Mul(c)                         // (batch, hidden)
		b := i.Mul(g)                         // (batch, hidden)
		c = a.Add(b)                          // (batch, hidden)
		h = o.Mul(c.Tanh())                   // (batch, hidden)
		if result == nil {
			result = h
		} else {
			result = tensor.VStack(result, h)
		}
	}
	return result.Reshape(inputShape[0], inputShape[1], int64(layer.hidden)),
		copyState(h), copyState(c)
}

func (layer *Lstm) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
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
