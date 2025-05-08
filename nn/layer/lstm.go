package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
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

func NewLstm(name string, featureSize, steps, hidden int, opts ...LayerCreateOption) *Lstm {
	var layer Lstm
	layer.new("lstm", name, opts...)
	layer.featureSize = featureSize
	layer.steps = steps
	layer.hidden = hidden
	layer.Wi = layer.initW(int64(featureSize+hidden), int64(hidden))
	layer.Wf = layer.initW(int64(featureSize+hidden), int64(hidden))
	layer.Wg = layer.initW(int64(featureSize+hidden), int64(hidden))
	layer.Wo = layer.initW(int64(featureSize+hidden), int64(hidden))
	layer.Bi = layer.initB(int64(hidden))
	layer.Bf = layer.initB(int64(hidden))
	layer.Bg = layer.initB(int64(hidden))
	layer.Bo = layer.initB(int64(hidden))
	return &layer
}

func LoadLstm(name string, params []*tensor.Tensor, args map[string]float32) Layer {
	var layer Lstm
	layer.new("lstm", name)
	layer.featureSize = int(args["feature_size"])
	layer.steps = int(args["steps"])
	layer.hidden = int(args["hidden"])
	layer.Wi = params[0]
	layer.Wf = params[1]
	layer.Wg = params[2]
	layer.Wo = params[3]
	layer.Bi = params[4]
	layer.Bf = params[5]
	layer.Bg = params[6]
	layer.Bo = params[7]
	return &layer
}

func (layer *Lstm) Forward(x, h, c *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	inputShape := x.Shapes()
	if h == nil {
		h = tensor.Zeros(consts.KFloat,
			tensor.WithShapes(int64(inputShape[0]), int64(layer.hidden)),
			tensor.WithDevice(layer.device))
	}
	if c == nil {
		c = tensor.Zeros(consts.KFloat,
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
		copyState(h),
		copyState(c)
}

func (layer *Lstm) Params() []*tensor.Tensor {
	return []*tensor.Tensor{
		layer.Wi,
		layer.Wf,
		layer.Wg,
		layer.Wo,
		layer.Bi,
		layer.Bf,
		layer.Bg,
		layer.Bo,
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
	layer.Wi.SetRequiresGrad(false)
	layer.Wf.SetRequiresGrad(false)
	layer.Wg.SetRequiresGrad(false)
	layer.Wo.SetRequiresGrad(false)
	layer.Bi.SetRequiresGrad(false)
	layer.Bf.SetRequiresGrad(false)
	layer.Bg.SetRequiresGrad(false)
	layer.Bo.SetRequiresGrad(false)
}

func (layer *Lstm) Unfreeze() {
	layer.Wi.SetRequiresGrad(true)
	layer.Wf.SetRequiresGrad(true)
	layer.Wg.SetRequiresGrad(true)
	layer.Wo.SetRequiresGrad(true)
	layer.Bi.SetRequiresGrad(true)
	layer.Bf.SetRequiresGrad(true)
	layer.Bg.SetRequiresGrad(true)
	layer.Bo.SetRequiresGrad(true)
}

func (layer *Lstm) ToScalarType(t consts.ScalarType) {
	layer.Wi = layer.Wi.ToScalarType(t)
	layer.Wf = layer.Wf.ToScalarType(t)
	layer.Wg = layer.Wg.ToScalarType(t)
	layer.Wo = layer.Wo.ToScalarType(t)
	layer.Bi = layer.Bi.ToScalarType(t)
	layer.Bf = layer.Bf.ToScalarType(t)
	layer.Bg = layer.Bg.ToScalarType(t)
	layer.Bo = layer.Bo.ToScalarType(t)
}

func (layer *Lstm) Reset() {
	layer.Wi = layer.initW(layer.Wi.Shapes()...)
	layer.Wf = layer.initW(layer.Wf.Shapes()...)
	layer.Wg = layer.initW(layer.Wg.Shapes()...)
	layer.Wo = layer.initW(layer.Wo.Shapes()...)
	layer.Bi = layer.initB(layer.Bi.Shapes()...)
	layer.Bf = layer.initB(layer.Bf.Shapes()...)
	layer.Bg = layer.initB(layer.Bg.Shapes()...)
	layer.Bo = layer.initB(layer.Bo.Shapes()...)
}

func (layer *Lstm) Clone() Layer {
	return &Lstm{
		base:        layer.base.clone(),
		featureSize: layer.featureSize,
		steps:       layer.steps,
		hidden:      layer.hidden,
		Wi:          layer.Wi.Clone(),
		Wf:          layer.Wf.Clone(),
		Wg:          layer.Wg.Clone(),
		Wo:          layer.Wo.Clone(),
		Bi:          layer.Bi.Clone(),
		Bf:          layer.Bf.Clone(),
		Bg:          layer.Bg.Clone(),
		Bo:          layer.Bo.Clone(),
	}
}

func (layer *Lstm) Update(params []*tensor.Tensor) {
	layer.Wi = params[0]
	layer.Wf = params[1]
	layer.Wg = params[2]
	layer.Wo = params[3]
	layer.Bi = params[4]
	layer.Bf = params[5]
	layer.Bg = params[6]
	layer.Bo = params[7]
}
