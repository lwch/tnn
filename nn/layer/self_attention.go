package layer

import (
	"math"

	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type SelfAttention struct {
	*base
	steps, dims, heads int
	dropout            float64
	scale              *tensor.Tensor
	// params
	wq, wk, wv *tensor.Tensor
	bq, bk, bv *tensor.Tensor
}

func NewSelfAttention(steps, dims, heads int, dropout float64) *SelfAttention {
	var layer SelfAttention
	layer.base = new("self_attention")
	layer.steps = steps
	layer.dims = dims
	layer.heads = heads
	layer.dropout = dropout
	if layer.dims%layer.heads != 0 {
		panic("dims must be divisible by heads")
	}
	return &layer
}

func LoadSelfAttention(name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer SelfAttention
	layer.base = new("self_attention")
	layer.name = name
	layer.steps = int(args["steps"])
	layer.dims = int(args["dims"])
	layer.heads = int(args["heads"])
	layer.dropout = float64(args["dropout"])
	layer.wq = loadParam(params["Wq"])
	layer.wk = loadParam(params["Wk"])
	layer.wv = loadParam(params["Wv"])
	layer.bq = loadParam(params["Bq"])
	layer.bk = loadParam(params["Bk"])
	layer.bv = loadParam(params["Bv"])
	return &layer
}

func (layer *SelfAttention) Forward(q, k, mask *tensor.Tensor, train bool) *tensor.Tensor {
	if layer.scale == nil {
		layer.scale = tensor.FromFloat32(nil, []float32{float32(math.Sqrt(float64(layer.dims)))}, 1)
	}
	if layer.wq == nil {
		layer.wq = initW(int64(layer.dims), int64(layer.dims))
	}
	if layer.wk == nil {
		layer.wk = initW(int64(layer.dims), int64(layer.dims))
	}
	if layer.wv == nil {
		layer.wv = initW(int64(layer.dims), int64(layer.dims))
	}
	if layer.bq == nil {
		layer.bq = initB(int64(layer.steps), int64(layer.dims))
	}
	if layer.bk == nil {
		layer.bk = initB(int64(layer.steps), int64(layer.dims))
	}
	if layer.bv == nil {
		layer.bv = initB(int64(layer.steps), int64(layer.dims))
	}
	q = q.MatMul(layer.wq).Add(layer.bq)  // (batch, steps, dims)
	k = k.MatMul(layer.wk).Add(layer.bk)  // (batch, steps, dims)
	v := k.MatMul(layer.wv).Add(layer.bv) // (batch, steps, dims)
	q = layer.split(q)                    // (batch, heads, steps, dims/heads)
	k = layer.split(k)                    // (batch, heads, steps, dims/heads)
	v = layer.split(v)                    // (batch, heads, steps, dims/heads)
	y := q.MatMul(k.Transpose(-1, -2))    // (batch, heads, steps, steps)
	y = y.Div(layer.scale)                // (batch, heads, steps, steps)
	if mask != nil {
		y = y.Add(mask) // (batch, heads, steps, steps)
	}
	y = y.Softmax(-1)                                                   // (batch, heads, steps, steps)
	y = y.Dropout(layer.dropout, train)                                 // (batch, heads, steps, steps)
	y = y.MatMul(v)                                                     // (batch, heads, steps, dims/heads)
	y = y.Permute(0, 2, 1, 3)                                           // (batch, steps, heads, dims/heads)
	y = y.Reshape(y.Shapes()[0], int64(layer.steps), int64(layer.dims)) // (batch, steps, dims)
	return y
}

func (layer *SelfAttention) split(x *tensor.Tensor) *tensor.Tensor {
	inputShape := x.Shapes()
	v := x.View(inputShape[0], int64(layer.steps), int64(layer.heads), int64(layer.dims/layer.heads))
	return v.Permute(0, 2, 1, 3)
}

func (layer *SelfAttention) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"Wq": layer.wq, "Wk": layer.wk, "Wv": layer.wv,
		"Bq": layer.bq, "Bk": layer.bk, "Bv": layer.bv,
	}
}

func (layer *SelfAttention) Args() map[string]float32 {
	return map[string]float32{
		"steps":   float32(layer.steps),
		"dims":    float32(layer.dims),
		"heads":   float32(layer.heads),
		"dropout": float32(layer.dropout),
	}
}
