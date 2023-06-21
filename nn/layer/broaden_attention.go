package layer

import (
	"math"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type BroadenAttention struct {
	*base
	steps, dims, multiple, heads int
	dropout                      float64
	scale                        *tensor.Tensor
	// params
	wq, wk, wv *tensor.Tensor
	bq, bk, bv *tensor.Tensor
}

func NewBroadenAttention(steps, dims, multiple, heads int, dropout float64, device consts.DeviceType) *BroadenAttention {
	var layer BroadenAttention
	layer.base = new("broaden_attention", device)
	layer.steps = steps
	layer.dims = dims
	layer.multiple = multiple
	layer.heads = heads
	layer.dropout = dropout
	if layer.dims%layer.heads != 0 {
		panic("dims must be divisible by heads")
	}
	return &layer
}

func LoadBroadenAttention(device consts.DeviceType, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer BroadenAttention
	layer.base = new("broaden_attention", device)
	layer.name = name
	layer.steps = int(args["steps"])
	layer.dims = int(args["dims"])
	layer.multiple = int(args["multiple"])
	layer.heads = int(args["heads"])
	layer.dropout = float64(args["dropout"])
	layer.wq = layer.loadParam(params["Wq"])
	layer.wk = layer.loadParam(params["Wk"])
	layer.wv = layer.loadParam(params["Wv"])
	layer.bq = layer.loadParam(params["Bq"])
	layer.bk = layer.loadParam(params["Bk"])
	layer.bv = layer.loadParam(params["Bv"])
	return &layer
}

func (layer *BroadenAttention) Forward(q, k, mask *tensor.Tensor, train bool) *tensor.Tensor {
	if layer.scale == nil {
		layer.scale = tensor.FromFloat32(nil, []float32{float32(math.Sqrt(float64(layer.dims)))}, tensor.WithShapes(1))
	}
	if layer.wq == nil {
		layer.wq = layer.initW(int64(layer.dims), int64(layer.dims*layer.multiple))
	}
	if layer.wk == nil {
		layer.wk = layer.initW(int64(layer.dims), int64(layer.dims*layer.multiple))
	}
	if layer.wv == nil {
		layer.wv = layer.initW(int64(layer.dims), int64(layer.dims))
	}
	if layer.bq == nil {
		layer.bq = layer.initB(int64(layer.steps), int64(layer.dims*layer.multiple))
	}
	if layer.bk == nil {
		layer.bk = layer.initB(int64(layer.steps), int64(layer.dims*layer.multiple))
	}
	if layer.bv == nil {
		layer.bv = layer.initB(int64(layer.steps), int64(layer.dims))
	}
	q = q.MatMul(layer.wq).Add(layer.bq)  // (batch, steps, dims*multiple)
	v := k.MatMul(layer.wv).Add(layer.bv) // (batch, steps, dims)
	k = k.MatMul(layer.wk).Add(layer.bk)  // (batch, steps, dims*multiple)
	q = layer.split(q, layer.multiple)    // (batch, heads*multiple, steps, dims/heads)
	k = layer.split(k, layer.multiple)    // (batch, heads*multiple, steps, dims/heads)
	v = layer.split(v, 1)                 // (batch, heads, steps, dims/heads)
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

func (layer *BroadenAttention) split(x *tensor.Tensor, multiple int) *tensor.Tensor {
	inputShape := x.Shapes()
	v := x.View(inputShape[0], int64(layer.steps), int64(layer.heads*multiple), int64(layer.dims/layer.heads))
	return v.Permute(0, 2, 1, 3)
}

func (layer *BroadenAttention) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"Wq": layer.wq, "Wk": layer.wk, "Wv": layer.wv,
		"Bq": layer.bq, "Bk": layer.bk, "Bv": layer.bv,
	}
}

func (layer *BroadenAttention) Args() map[string]float32 {
	return map[string]float32{
		"steps":    float32(layer.steps),
		"dims":     float32(layer.dims),
		"multiple": float32(layer.multiple),
		"heads":    float32(layer.heads),
		"dropout":  float32(layer.dropout),
	}
}
