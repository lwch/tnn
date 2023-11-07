package layer

import (
	"math"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

type Attention1 struct {
	base
	dims, heads int
	dropout     float64
	// params
	w     *tensor.Tensor
	scale *tensor.Tensor
}

func NewAttention1(dims, heads int, dropout float64, opts ...LayerCreateOption) *Attention1 {
	var layer Attention1
	layer.new("attention1", opts...)
	layer.dims = dims
	layer.heads = heads
	layer.dropout = dropout
	if layer.dims%layer.heads != 0 {
		panic("dims must be divisible by heads")
	}
	layer.scale = tensor.FromFloat32(nil, []float32{float32(math.Sqrt(float64(dims)))},
		tensor.WithShapes(1),
		tensor.WithDevice(layer.device))
	layer.w = layer.initW(int64(dims*3), int64(dims*3))
	return &layer
}

func LoadAttention1(name string, params map[string]*tensor.Tensor, args map[string]float32) Layer {
	var layer Attention1
	layer.new("attention1")
	layer.name = name
	layer.dims = int(args["dims"])
	layer.heads = int(args["heads"])
	layer.dropout = float64(args["dropout"])
	layer.w = params["w"]
	layer.scale = tensor.FromFloat32(nil, []float32{float32(math.Sqrt(float64(layer.dims)))},
		tensor.WithShapes(1),
		tensor.WithDevice(layer.device))
	return &layer
}

func (layer *Attention1) Forward(q, k, v, mask *tensor.Tensor, isCausal, train bool) *tensor.Tensor {
	if mask != nil && isCausal {
		panic("unexpected mask")
	}
	inputShape := q.Shapes()
	x := tensor.Cat([]*tensor.Tensor{q, k, v}, -1)           // (batch, seq, dims*3)
	x = x.MatMul(layer.w.Transpose(0, 1))                    // (batch, seq, dims*3)
	q = x.NArrow(-1, 0, int64(layer.dims))                   // (batch, seq, dims)
	k = x.NArrow(-1, int64(layer.dims), int64(layer.dims))   // (batch, seq, dims)
	v = x.NArrow(-1, int64(layer.dims*2), int64(layer.dims)) // (batch, seq, dims)
	q = layer.split(q)                                       // (batch, heads, seq, dims/heads)
	k = layer.split(k)                                       // (batch, heads, seq, dims/heads)
	v = layer.split(v)                                       // (batch, heads, seq, dims/heads)
	if isCausal {
		mask = buildCausal(q, k, layer.device)
	}
	score := q.MatMul(k.Transpose(-2, -1)).Div(layer.scale) // (batch, heads, seq, dims/heads)
	if mask != nil {
		score = score.Add(mask) // (batch, heads, seq, dims/heads)
	}
	score = score.Softmax1(-1)                          // (batch, heads, seq, dims/heads)
	y := score.Dropout(layer.dropout, train).MatMul(v)  // (batch, heads, seq, dims/heads)
	y = y.Transpose(1, 2)                               // (batch, seq, heads, dims/heads)
	y = y.Reshape(-1, inputShape[1], int64(layer.dims)) // (batch, seq, dims)
	return y
}

func (layer *Attention1) Score(q, k, v, mask *tensor.Tensor, isCausal, train bool) *tensor.Tensor {
	if mask != nil && isCausal {
		panic("unexpected mask")
	}
	x := tensor.Cat([]*tensor.Tensor{q, k, v}, -1)         // (batch, seq, dims*3)
	x = x.MatMul(layer.w)                                  // (batch, seq, dims*3)
	q = x.NArrow(-1, 0, int64(layer.dims))                 // (batch, seq, dims)
	k = x.NArrow(-1, int64(layer.dims), int64(layer.dims)) // (batch, seq, dims)
	q = layer.split(q)                                     // (batch, heads, seq, dims/heads)
	k = layer.split(k)                                     // (batch, heads, seq, dims/heads)
	if isCausal {
		mask = buildCausal(q, k, layer.device)
	}
	score := q.MatMul(k.Transpose(-2, -1)).Div(layer.scale) // (batch, heads, seq, dims/heads)
	if mask != nil {
		score = score.Add(mask) // (batch, heads, seq, dims/heads)
	}
	return score.Softmax1(-1) // (batch, heads, seq, dims/heads)
}

func (layer *Attention1) split(x *tensor.Tensor) *tensor.Tensor {
	y := x.View(-1, x.Shapes()[1], int64(layer.heads), int64(layer.dims/layer.heads))
	return y.Transpose(1, 2)
}

func buildCausal(q, k *tensor.Tensor, device consts.DeviceType) *tensor.Tensor {
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
	storage := q.Storage()
	if storage == nil {
		storage = k.Storage()
	}
	return tensor.FromFloat32(storage, mask,
		tensor.WithShapes(1, 1, l, s),
		tensor.WithDevice(device))
}

func (layer *Attention1) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": layer.w,
	}
}

func (layer *Attention1) Args() map[string]float32 {
	return map[string]float32{
		"dims":    float32(layer.dims),
		"heads":   float32(layer.heads),
		"dropout": float32(layer.dropout),
	}
}

func (layer *Attention1) Freeze() {
	layer.w.SetRequiresGrad(false)
}

func (layer *Attention1) Unfreeze() {
	layer.w.SetRequiresGrad(true)
}
