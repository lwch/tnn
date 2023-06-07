package layer

import (
	"math"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/internal/pb"
	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

type SelfAttention struct {
	*base
	steps, dims int
	scale       *ts.Tensor
	// params
	wq, wk, wv *ts.Tensor
	bq, bk, bv *ts.Tensor
}

func NewSelfAttention(steps, dims int) *SelfAttention {
	var layer SelfAttention
	layer.base = new("self_attention")
	layer.steps = steps
	layer.dims = dims
	return &layer
}

func LoadSelfAttention(vs *nn.Path, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer SelfAttention
	layer.base = new("self_attention")
	layer.name = name
	layer.steps = int(args["steps"])
	layer.dims = int(args["dims"])
	layer.wq = loadParam(vs, params["Wq"], "Wq")
	layer.wk = loadParam(vs, params["Wk"], "Wk")
	layer.wv = loadParam(vs, params["Wv"], "Wv")
	layer.bq = loadParam(vs, params["Bq"], "Bq")
	layer.bk = loadParam(vs, params["Bk"], "Bk")
	layer.bv = loadParam(vs, params["Bv"], "Bv")
	return &layer
}

func (layer *SelfAttention) Forward(vs *nn.Path, x *ts.Tensor) *ts.Tensor {
	inputShape := x.MustSize()
	if layer.scale == nil {
		scale, err := ts.NewTensorFromData(
			float32(math.Sqrt(float64(layer.dims))),
			[]int64{1})
		runtime.Assert(err)
		layer.scale = vs.MustAdd("scale", scale, false)
	}
	if layer.wq == nil {
		layer.wq = initW(vs, "Wq", int64(layer.dims), int64(layer.dims))
	}
	if layer.wk == nil {
		layer.wk = initW(vs, "Wk", int64(layer.dims), int64(layer.dims))
	}
	if layer.wv == nil {
		layer.wv = initW(vs, "Wv", int64(layer.dims), int64(layer.dims))
	}
	if layer.bq == nil {
		layer.bq = initB(vs, "Bq", int64(layer.steps), int64(layer.dims))
	}
	if layer.bk == nil {
		layer.bk = initB(vs, "Bk", int64(layer.steps), int64(layer.dims))
	}
	if layer.bv == nil {
		layer.bv = initB(vs, "Bv", int64(layer.steps), int64(layer.dims))
	}
	var result *ts.Tensor
	for batch := int64(0); batch < inputShape[0]; batch++ {
		x := x.MustNarrow(0, int64(batch), 1, false).
			MustReshape([]int64{int64(layer.steps), int64(layer.dims)}, false) // (steps, dims)
		q := x.MustMm(layer.wq, false).MustAdd(layer.bq, false) // (steps, dims)
		k := x.MustMm(layer.wk, false).MustAdd(layer.bk, false) // (steps, dims)
		v := x.MustMm(layer.wv, false).MustAdd(layer.bv, false) // (steps, dims)
		a := q.MustMm(k.MustTranspose(1, 0, false), false)      // (steps, steps)
		a = a.MustDiv(layer.scale, false)                       // (steps, steps)
		a = a.MustSoftmax(1, gotch.Float, false)                // (steps, steps)
		a = a.MustMm(v, false)                                  // (steps, dims
		if result == nil {
			result = a
		} else {
			result = ts.MustVstack([]ts.Tensor{*result, *a})
		}
	}
	return result.MustReshape([]int64{inputShape[0], int64(layer.steps), int64(layer.dims)}, true)
}

func (layer *SelfAttention) Params() map[string]*ts.Tensor {
	return map[string]*ts.Tensor{
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
