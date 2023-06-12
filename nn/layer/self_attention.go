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

func (layer *SelfAttention) Forward(x *tensor.Tensor) *tensor.Tensor {
	inputShape := x.Shapes()
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
	// TODO
	return x
	// var result *tensor.Tensor
	// for batch := int64(0); batch < inputShape[0]; batch++ {
	// 	x := x.MustNarrow(0, int64(batch), 1, false).
	// 		MustReshape([]int64{int64(layer.steps), int64(layer.dims)}, false) // (steps, dims)
	// 	q := x.MustMm(layer.wq, false).MustAdd(layer.bq, false) // (steps, dims)
	// 	k := x.MustMm(layer.wk, false).MustAdd(layer.bk, false) // (steps, dims)
	// 	v := x.MustMm(layer.wv, false).MustAdd(layer.bv, false) // (steps, dims)
	// 	a := q.MustMm(k.MustTranspose(1, 0, false), false)      // (steps, steps)
	// 	a = a.MustDiv(layer.scale, false)                       // (steps, steps)
	// 	a = a.MustSoftmax(1, gotch.Float, false)                // (steps, steps)
	// 	a = a.MustMm(v, false)                                  // (steps, dims
	// 	if result == nil {
	// 		result = a
	// 	} else {
	// 		result = ts.MustVstack([]ts.Tensor{*result, *a})
	// 	}
	// }
	// return result.MustReshape([]int64{inputShape[0], int64(layer.steps), int64(layer.dims)}, true)
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
