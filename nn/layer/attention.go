package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type Attention struct {
	base
	dims, heads int
	dropout     float64
	isCausal    bool
	// params
	wq, wk, wv *tensor.Tensor
	bq, bk, bv *tensor.Tensor
}

func NewAttention(dims, heads int, dropout float64, isCausal bool, opts ...LayerCreateOption) *Attention {
	var layer Attention
	layer.new("attention", opts...)
	layer.dims = dims
	layer.heads = heads
	layer.dropout = dropout
	layer.isCausal = isCausal
	if layer.dims%layer.heads != 0 {
		panic("dims must be divisible by heads")
	}
	return &layer
}

func LoadAttention(device consts.DeviceType, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Attention
	layer.new("attention", WithDevice(device))
	layer.name = name
	layer.dims = int(args["dims"])
	layer.heads = int(args["heads"])
	layer.dropout = float64(args["dropout"])
	layer.isCausal = args["is_causal"] != 0
	layer.wq = layer.loadParam(params["Wq"])
	layer.wk = layer.loadParam(params["Wk"])
	layer.wv = layer.loadParam(params["Wv"])
	layer.bq = layer.loadParam(params["Bq"])
	layer.bk = layer.loadParam(params["Bk"])
	layer.bv = layer.loadParam(params["Bv"])
	return &layer
}

func seqLen(t *tensor.Tensor) []int64 {
	shapes := t.Shapes()
	ret := make([]int64, 0, len(shapes)-2)
	for i := 1; i < len(shapes)-1; i++ {
		ret = append(ret, shapes[i])
	}
	return ret
}

func (layer *Attention) Forward(q, k, v, mask *tensor.Tensor, train bool) (*tensor.Tensor, *tensor.Tensor) {
	if mask != nil && layer.isCausal {
		panic("unexpected mask")
	}
	inputShape := q.Shapes()
	if layer.wq == nil {
		layer.wq = layer.initW(int64(layer.dims), int64(layer.dims))
	}
	if layer.wk == nil {
		layer.wk = layer.initW(int64(layer.dims), int64(layer.dims))
	}
	if layer.wv == nil {
		layer.wv = layer.initW(int64(layer.dims), int64(layer.dims))
	}
	if layer.bq == nil {
		layer.bq = layer.initB(int64(layer.dims))
	}
	if layer.bk == nil {
		layer.bk = layer.initB(int64(layer.dims))
	}
	if layer.bv == nil {
		layer.bv = layer.initB(int64(layer.dims))
	}
	conv1d := func(x, w, b *tensor.Tensor, dims int64) *tensor.Tensor {
		shapes := x.Shapes()
		x = x.View(-1, dims).MatMul(w).Add(b)
		return x.View(shapes...)
	}
	q = conv1d(q, layer.wq, layer.bq, int64(layer.dims)) // (batch, ..., dims)
	k = conv1d(k, layer.wk, layer.bk, int64(layer.dims)) // (batch, ..., dims)
	v = conv1d(v, layer.wv, layer.bv, int64(layer.dims)) // (batch, ..., dims)
	q = layer.split(q)                                   // (batch, heads, ..., dims/heads)
	k = layer.split(k)                                   // (batch, heads, ..., dims/heads)
	v = layer.split(v)                                   // (batch, heads, ..., dims/heads)
	dropout := layer.dropout
	if !train {
		dropout = 0
	}
	y, score := tensor.ScaledDotProductAttention(q, k, v, mask, dropout, layer.isCausal) // (batch, heads, ..., dims/heads)
	idx := make([]int64, len(inputShape)+1)
	idx[0] = 0
	for i := 1; i < len(inputShape)-1; i++ {
		idx[i] = int64(i + 1)
	}
	idx[len(idx)-2] = 1
	idx[len(idx)-1] = -1
	y = y.Permute(idx...).Contiguous() // (batch, ..., heads, dims/heads)
	y = y.View(inputShape...)          // (batch, ..., dims)
	return y, score
}

func (layer *Attention) split(x *tensor.Tensor) *tensor.Tensor {
	inputShape := x.Shapes()
	dims := make([]int64, len(inputShape)+1)
	idx := make([]int64, len(inputShape)+1)
	dims[0] = inputShape[0]
	idx[0] = 0
	idx[1] = -2
	for i, dim := range seqLen(x) {
		dims[i+1] = dim
		idx[i+2] = int64(i + 1)
	}
	idx[len(idx)-1] = -1
	dims[len(dims)-2] = int64(layer.heads)
	dims[len(dims)-1] = int64(layer.dims / layer.heads)
	v := x.View(dims...)
	return v.Permute(idx...)
}

func (layer *Attention) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"Wq": layer.wq, "Wk": layer.wk, "Wv": layer.wv,
		"Bq": layer.bq, "Bk": layer.bk, "Bv": layer.bv,
	}
}

func (layer *Attention) Args() map[string]float32 {
	var isCausal float32
	if layer.isCausal {
		isCausal = 1
	}
	return map[string]float32{
		"dims":      float32(layer.dims),
		"heads":     float32(layer.heads),
		"dropout":   float32(layer.dropout),
		"is_causal": isCausal,
	}
}

func (layer *Attention) Freeze() {
	if layer.wq != nil {
		layer.wq.SetRequiresGrad(false)
	}
	if layer.wk != nil {
		layer.wk.SetRequiresGrad(false)
	}
	if layer.wv != nil {
		layer.wv.SetRequiresGrad(false)
	}
	if layer.bq != nil {
		layer.bq.SetRequiresGrad(false)
	}
	if layer.bk != nil {
		layer.bk.SetRequiresGrad(false)
	}
	if layer.bv != nil {
		layer.bv.SetRequiresGrad(false)
	}
}

func (layer *Attention) Unfreeze() {
	if layer.wq != nil {
		layer.wq.SetRequiresGrad(true)
	}
	if layer.wk != nil {
		layer.wk.SetRequiresGrad(true)
	}
	if layer.wv != nil {
		layer.wv.SetRequiresGrad(true)
	}
	if layer.bq != nil {
		layer.bq.SetRequiresGrad(true)
	}
	if layer.bk != nil {
		layer.bk.SetRequiresGrad(true)
	}
	if layer.bv != nil {
		layer.bv.SetRequiresGrad(true)
	}
}
