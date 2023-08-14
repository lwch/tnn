package layer

import (
	"math"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type Attention1 struct {
	base
	dims, heads int
	dropout     float64
	isCausal    bool
	// params
	wq, wk, wv, wo *tensor.Tensor
	bq, bk, bv, bo *tensor.Tensor
	scale          *tensor.Tensor
}

func NewAttention1(dims, heads int, dropout float64, isCausal bool, opts ...LayerCreateOption) *Attention1 {
	var layer Attention1
	layer.new("attention1", opts...)
	layer.dims = dims
	layer.heads = heads
	layer.dropout = dropout
	layer.isCausal = isCausal
	if layer.dims%layer.heads != 0 {
		panic("dims must be divisible by heads")
	}
	layer.scale = tensor.FromFloat32(nil, []float32{float32(math.Sqrt(float64(dims)))},
		tensor.WithShapes(1),
		tensor.WithDevice(layer.device))
	return &layer
}

func LoadAttention1(device consts.DeviceType, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Attention1
	layer.new("attention1", WithDevice(device))
	layer.name = name
	layer.dims = int(args["dims"])
	layer.heads = int(args["heads"])
	layer.dropout = float64(args["dropout"])
	layer.isCausal = args["is_causal"] != 0
	layer.wq = layer.loadParam(params["Wq"])
	layer.wk = layer.loadParam(params["Wk"])
	layer.wv = layer.loadParam(params["Wv"])
	layer.wo = layer.loadParam(params["Wo"])
	layer.bq = layer.loadParam(params["Bq"])
	layer.bk = layer.loadParam(params["Bk"])
	layer.bv = layer.loadParam(params["Bv"])
	layer.bo = layer.loadParam(params["Bo"])
	layer.scale = tensor.FromFloat32(nil, []float32{float32(math.Sqrt(float64(layer.dims)))},
		tensor.WithShapes(1),
		tensor.WithDevice(layer.device))
	return &layer
}

func (layer *Attention1) Forward(q, k, v, mask *tensor.Tensor, train bool) (*tensor.Tensor, *tensor.Tensor) {
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
	if layer.wo == nil {
		layer.wo = layer.initW(int64(layer.dims), int64(layer.dims))
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
	if layer.bo == nil {
		layer.bo = layer.initB(int64(layer.dims))
	}
	q = q.MatMul(layer.wq).Add(layer.bq) // (batch, ..., dims)
	k = k.MatMul(layer.wk).Add(layer.bk) // (batch, ..., dims)
	v = v.MatMul(layer.wv).Add(layer.bv) // (batch, ..., dims)
	q = layer.split(q)                   // (batch, heads, ..., dims/heads)
	k = layer.split(k)                   // (batch, heads, ..., dims/heads)
	v = layer.split(v)                   // (batch, heads, ..., dims/heads)
	if layer.isCausal {
		mask = layer.buildCausal(q, k)
	}
	score := q.MatMul(k.Transpose(-1, -2)).Div(layer.scale) // (batch, heads, ..., dims/heads)
	if mask != nil {
		score = score.Add(mask) // (batch, heads, ..., dims/heads)
	}
	score = score.Softmax1(-1)                  // (batch, heads, ..., dims/heads)
	score = score.Dropout(layer.dropout, train) // (batch, heads, ..., dims/heads)
	y := score.MatMul(v)                        // (batch, heads, ..., dims/heads)
	idx := make([]int64, len(inputShape)+1)
	idx[0] = 0
	for i := 1; i < len(inputShape)-1; i++ {
		idx[i] = int64(i + 1)
	}
	idx[len(idx)-2] = 1
	idx[len(idx)-1] = -1
	y = y.Permute(idx...).Contiguous()   // (batch, ..., heads, dims/heads)
	y = y.View(inputShape...)            // (batch, ..., dims)
	y = y.MatMul(layer.wo).Add(layer.bo) // (batch, ..., dims)
	return y, score
}

func (layer *Attention1) split(x *tensor.Tensor) *tensor.Tensor {
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

func (layer *Attention1) buildCausal(q, k *tensor.Tensor) *tensor.Tensor {
	l := q.Shapes()[q.Dims()-2]
	s := k.Shapes()[k.Dims()-2]
	mask := make([]float32, l*s)
	for i := int64(0); i < l; i++ {
		for j := int64(0); j < s; j++ {
			if j > i {
				mask[i*s+j] = float32(math.Inf(-1))
			}
		}
	}
	dims := make([]int64, 0, q.Dims())
	for i := int64(0); i < q.Dims()-2; i++ {
		dims = append(dims, 1)
	}
	dims = append(dims, l, s)
	storage := q.Storage()
	if storage == nil {
		storage = k.Storage()
	}
	return tensor.FromFloat32(storage, mask,
		tensor.WithShapes(dims...),
		tensor.WithDevice(layer.device))
}

func (layer *Attention1) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"Wq": layer.wq, "Wk": layer.wk, "Wv": layer.wv, "Wo": layer.wo,
		"Bq": layer.bq, "Bk": layer.bk, "Bv": layer.bv, "Bo": layer.bo,
	}
}

func (layer *Attention1) Args() map[string]float32 {
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

func (layer *Attention1) Freeze() {
	if layer.wq != nil {
		layer.wq.SetRequiresGrad(false)
	}
	if layer.wk != nil {
		layer.wk.SetRequiresGrad(false)
	}
	if layer.wv != nil {
		layer.wv.SetRequiresGrad(false)
	}
	if layer.wo != nil {
		layer.wo.SetRequiresGrad(false)
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
	if layer.bo != nil {
		layer.bo.SetRequiresGrad(false)
	}
}

func (layer *Attention1) Unfreeze() {
	if layer.wq != nil {
		layer.wq.SetRequiresGrad(true)
	}
	if layer.wk != nil {
		layer.wk.SetRequiresGrad(true)
	}
	if layer.wv != nil {
		layer.wv.SetRequiresGrad(true)
	}
	if layer.wo != nil {
		layer.wo.SetRequiresGrad(true)
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
	if layer.bo != nil {
		layer.bo.SetRequiresGrad(true)
	}
}
