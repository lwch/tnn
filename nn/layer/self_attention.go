package layer

import (
	"math"

	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type SelfAttention struct {
	*base
	steps, dims int
	scale       *tensor.Tensor
	// params
	wq, wk, wv *tensor.Tensor
	bq, bk, bv *tensor.Tensor
}

func NewSelfAttention(steps, dims int) *SelfAttention {
	var layer SelfAttention
	layer.base = new("self_attention")
	layer.steps = steps
	layer.dims = dims
	return &layer
}

func LoadSelfAttention(name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer SelfAttention
	layer.base = new("self_attention")
	layer.name = name
	layer.steps = int(args["steps"])
	layer.dims = int(args["dims"])
	layer.wq = loadParam(params["Wq"])
	layer.wk = loadParam(params["Wk"])
	layer.wv = loadParam(params["Wv"])
	layer.bq = loadParam(params["Bq"])
	layer.bk = loadParam(params["Bk"])
	layer.bv = loadParam(params["Bv"])
	return &layer
}

func (layer *SelfAttention) Forward(x, mask *tensor.Tensor) *tensor.Tensor {
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
	q := x.MatMul(layer.wq).Add(layer.bq) // (batch, steps, dims)
	k := x.MatMul(layer.wk).Add(layer.bk) // (batch, steps, dims)
	v := x.MatMul(layer.wv).Add(layer.bv) // (batch, steps, dims)
	y := q.MatMul(k.Transpose(2, 1))      // (batch, steps, steps)
	y = y.Div(layer.scale)                // (batch, steps, steps)
	if mask != nil {
		y = y.Add(mask) // (batch, steps, steps)
	}
	y = y.Softmax(-1) // (batch, steps, steps)
	y = y.MatMul(v)   // (batch, steps, dims)
	return y
}

func (layer *SelfAttention) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"Wq": layer.wq, "Wk": layer.wk, "Wv": layer.wv,
		"Bq": layer.bq, "Bk": layer.bk, "Bv": layer.bv,
	}
}

func (layer *SelfAttention) Args() map[string]float32 {
	return map[string]float32{
		"steps": float32(layer.steps),
		"dims":  float32(layer.dims),
	}
}
