package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

type Attention struct {
	base
	dims, heads int
	dropout     float64
	// params
	w *tensor.Tensor
	b *tensor.Tensor
}

func NewAttention(dims, heads int, dropout float64, opts ...LayerCreateOption) *Attention {
	var layer Attention
	layer.new("attention", opts...)
	layer.dims = dims
	layer.heads = heads
	layer.dropout = dropout
	if layer.dims%layer.heads != 0 {
		panic("dims must be divisible by heads")
	}
	layer.w = layer.initW(int64(dims*3), int64(dims*3))
	layer.b = layer.initB(int64(dims * 3))
	return &layer
}

func LoadAttention(device consts.DeviceType, name string, params map[string]*tensor.Tensor, args map[string]float32) Layer {
	var layer Attention
	layer.new("attention", WithDevice(device))
	layer.name = name
	layer.dims = int(args["dims"])
	layer.heads = int(args["heads"])
	layer.dropout = float64(args["dropout"])
	layer.w = params["w"]
	layer.b = params["b"]
	return &layer
}

func (layer *Attention) Forward(q, k, v, mask *tensor.Tensor, isCausal, train bool) (*tensor.Tensor, *tensor.Tensor) {
	if mask != nil && isCausal {
		panic("unexpected mask")
	}
	inputShape := q.Shapes()
	x := tensor.Cat([]*tensor.Tensor{q, k, v}, -1)           // (batch, seq, dims*3)
	x = x.MatMul(layer.w).Add(layer.b)                       // (batch, seq, dims*3)
	q = x.NArrow(-1, 0, int64(layer.dims))                   // (batch, seq, dims)
	k = x.NArrow(-1, int64(layer.dims), int64(layer.dims))   // (batch, seq, dims)
	v = x.NArrow(-1, int64(layer.dims*2), int64(layer.dims)) // (batch, seq, dims)
	q = layer.split(q)                                       // (batch, heads, seq, dims/heads)
	k = layer.split(k)                                       // (batch, heads, seq, dims/heads)
	v = layer.split(v)                                       // (batch, heads, seq, dims/heads)
	dropout := layer.dropout
	if !train {
		dropout = 0
	}
	y, score := tensor.ScaledDotProductAttention(q, k, v, mask, dropout, isCausal) // (batch, heads, seq, dims/heads)
	y = y.Transpose(1, 2)                                                          // (batch, seq, heads, dims/heads)
	y = y.Reshape(-1, inputShape[1], int64(layer.dims))                            // (batch, seq, dims)
	return y, score
}

func (layer *Attention) split(x *tensor.Tensor) *tensor.Tensor {
	y := x.View(-1, x.Shapes()[1], int64(layer.heads), int64(layer.dims/layer.heads))
	return y.Transpose(1, 2)
}

func (layer *Attention) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": layer.w,
		"b": layer.b,
	}
}

func (layer *Attention) Args() map[string]float32 {
	return map[string]float32{
		"dims":    float32(layer.dims),
		"heads":   float32(layer.heads),
		"dropout": float32(layer.dropout),
	}
}

func (layer *Attention) Freeze() {
	layer.w.SetRequiresGrad(false)
	layer.b.SetRequiresGrad(false)
}

func (layer *Attention) Unfreeze() {
	layer.w.SetRequiresGrad(true)
	layer.b.SetRequiresGrad(true)
}
