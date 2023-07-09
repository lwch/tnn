package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type SelfAttention struct {
	*base
	hidden, heads int
	dropout       float64
	isCausal      bool
	// params
	wq, wk, wv *tensor.Tensor
}

func NewSelfAttention(hidden, heads int, dropout float64, isCausal bool, device consts.DeviceType) *SelfAttention {
	var layer SelfAttention
	layer.base = new("self_attention", device)
	layer.hidden = hidden
	layer.heads = heads
	layer.dropout = dropout
	layer.isCausal = isCausal
	if layer.hidden%layer.heads != 0 {
		panic("dims must be divisible by heads")
	}
	return &layer
}

func LoadSelfAttention(device consts.DeviceType, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer SelfAttention
	layer.base = new("self_attention", device)
	layer.name = name
	layer.hidden = int(args["hidden"])
	layer.heads = int(args["heads"])
	layer.dropout = float64(args["dropout"])
	layer.isCausal = args["is_causal"] != 0
	layer.wq = layer.loadParam(params["Wq"])
	layer.wk = layer.loadParam(params["Wk"])
	layer.wv = layer.loadParam(params["Wv"])
	return &layer
}

func stepDim(t *tensor.Tensor) int64 {
	shapes := t.Shapes()
	return shapes[len(shapes)-2]
}

func lastDim(t *tensor.Tensor) int64 {
	shapes := t.Shapes()
	return shapes[len(shapes)-1]
}

func (layer *SelfAttention) Forward(q, k, v, mask *tensor.Tensor, train bool) *tensor.Tensor {
	inputShape := v.Shapes()
	if layer.wq == nil {
		layer.wq = layer.initW(lastDim(q), int64(layer.hidden))
	}
	if layer.wk == nil {
		layer.wk = layer.initW(lastDim(k), int64(layer.hidden))
	}
	if layer.wv == nil {
		layer.wv = layer.initW(lastDim(v), int64(layer.hidden))
	}
	q = q.MatMul(layer.wq) // (batch, steps, hidden)
	k = k.MatMul(layer.wk) // (batch, steps, hidden)
	v = v.MatMul(layer.wv) // (batch, steps, hidden)
	q = layer.split(q)     // (batch, heads, steps, hidden/heads)
	k = layer.split(k)     // (batch, heads, steps, hidden/heads)
	v = layer.split(v)     // (batch, heads, steps, hidden/heads)
	dropout := layer.dropout
	if !train {
		dropout = 0
	}
	y := tensor.ScaledDotProductAttention(q, k, v, mask, dropout, layer.isCausal) // (batch, heads, steps, hidden/heads)
	y = y.Permute(0, 2, 1, 3)                                                     // (batch, steps, heads, hidden/heads)
	y = y.Reshape(inputShape[0], inputShape[1], int64(layer.hidden))              // (batch, steps, hidden)
	return y
}

func (layer *SelfAttention) split(x *tensor.Tensor) *tensor.Tensor {
	inputShape := x.Shapes()
	v := x.View(inputShape[0], stepDim(x), int64(layer.heads), lastDim(x)/int64(layer.heads))
	return v.Permute(0, 2, 1, 3)
}

func (layer *SelfAttention) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"Wq": layer.wq, "Wk": layer.wk, "Wv": layer.wv,
	}
}

func (layer *SelfAttention) Args() map[string]float32 {
	var isCausal float32
	if layer.isCausal {
		isCausal = 1
	}
	return map[string]float32{
		"hidden":    float32(layer.hidden),
		"heads":     float32(layer.heads),
		"dropout":   float32(layer.dropout),
		"is_causal": isCausal,
	}
}

func (layer *SelfAttention) Freeze() {
	if layer.wq != nil {
		layer.wq.SetRequiresGrad(false)
	}
	if layer.wk != nil {
		layer.wk.SetRequiresGrad(false)
	}
	if layer.wv != nil {
		layer.wv.SetRequiresGrad(false)
	}
}

func (layer *SelfAttention) Unfreeze() {
	if layer.wq != nil {
		layer.wq.SetRequiresGrad(true)
	}
	if layer.wk != nil {
		layer.wk.SetRequiresGrad(true)
	}
	if layer.wv != nil {
		layer.wv.SetRequiresGrad(true)
	}
}
