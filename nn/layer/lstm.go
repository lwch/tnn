package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type Lstm struct {
	base
	featureSize, steps int
	hidden             int
	Wi, Bi             *tensor.Tensor
	Wf, Bf             *tensor.Tensor
	Wg, Bg             *tensor.Tensor
	Wo, Bo             *tensor.Tensor
}

func NewLstm(featureSize, steps, hidden int, opts ...LayerCreateOption) *Lstm {
	var layer Lstm
	layer.new("lstm", opts...)
	layer.featureSize = featureSize
	layer.steps = steps
	layer.hidden = hidden
	return &layer
}

func LoadLstm(device consts.DeviceType, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Lstm
	layer.new("lstm", WithDevice(device))
	layer.name = name
	layer.featureSize = int(args["feature_size"])
	layer.steps = int(args["steps"])
	layer.hidden = int(args["hidden"])
	layer.Wi = layer.loadParam(params["Wi"])
	layer.Wf = layer.loadParam(params["Wf"])
	layer.Wg = layer.loadParam(params["Wg"])
	layer.Wo = layer.loadParam(params["Wo"])
	layer.Bi = layer.loadParam(params["Bi"])
	layer.Bf = layer.loadParam(params["Bf"])
	layer.Bg = layer.loadParam(params["Bg"])
	layer.Bo = layer.loadParam(params["Bo"])
	return &layer
}

func (layer *Lstm) Forward(x, h, c *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	inputShape := x.Shapes()
	if layer.Wi == nil {
		layer.Wi = layer.initW(int64(layer.featureSize+layer.hidden), int64(layer.hidden))
	}
	if layer.Wf == nil {
		layer.Wf = layer.initW(int64(layer.featureSize+layer.hidden), int64(layer.hidden))
	}
	if layer.Wg == nil {
		layer.Wg = layer.initW(int64(layer.featureSize+layer.hidden), int64(layer.hidden))
	}
	if layer.Wo == nil {
		layer.Wo = layer.initW(int64(layer.featureSize+layer.hidden), int64(layer.hidden))
	}
	if layer.Bi == nil {
		layer.Bi = layer.initB(int64(layer.hidden))
	}
	if layer.Bf == nil {
		layer.Bf = layer.initB(int64(layer.hidden))
	}
	if layer.Bg == nil {
		layer.Bg = layer.initB(int64(layer.hidden))
	}
	if layer.Bo == nil {
		layer.Bo = layer.initB(int64(layer.hidden))
	}
	if h == nil {
		h = tensor.Zeros(nil, consts.KFloat,
			tensor.WithShapes(int64(inputShape[0]), int64(layer.hidden)),
			tensor.WithDevice(layer.device))
	}
	if c == nil {
		c = tensor.Zeros(nil, consts.KFloat,
			tensor.WithShapes(int64(inputShape[0]), int64(layer.hidden)),
			tensor.WithDevice(layer.device))
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

func (layer *Lstm) Freeze() {
	if layer.Wi != nil {
		layer.Wi.SetRequiresGrad(false)
	}
	if layer.Wf != nil {
		layer.Wf.SetRequiresGrad(false)
	}
	if layer.Wg != nil {
		layer.Wg.SetRequiresGrad(false)
	}
	if layer.Wo != nil {
		layer.Wo.SetRequiresGrad(false)
	}
	if layer.Bi != nil {
		layer.Bi.SetRequiresGrad(false)
	}
	if layer.Bf != nil {
		layer.Bf.SetRequiresGrad(false)
	}
	if layer.Bg != nil {
		layer.Bg.SetRequiresGrad(false)
	}
	if layer.Bo != nil {
		layer.Bo.SetRequiresGrad(false)
	}
}

func (layer *Lstm) Unfreeze() {
	if layer.Wi != nil {
		layer.Wi.SetRequiresGrad(true)
	}
	if layer.Wf != nil {
		layer.Wf.SetRequiresGrad(true)
	}
	if layer.Wg != nil {
		layer.Wg.SetRequiresGrad(true)
	}
	if layer.Wo != nil {
		layer.Wo.SetRequiresGrad(true)
	}
	if layer.Bi != nil {
		layer.Bi.SetRequiresGrad(true)
	}
	if layer.Bf != nil {
		layer.Bf.SetRequiresGrad(true)
	}
	if layer.Bg != nil {
		layer.Bg.SetRequiresGrad(true)
	}
	if layer.Bo != nil {
		layer.Bo.SetRequiresGrad(true)
	}
}
