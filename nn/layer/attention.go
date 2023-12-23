package layer

import (
	"math"

	"github.com/lwch/gotorch/tensor"
)

type Attention struct {
	base
	dims, heads int
	dropout     float64
	rope        bool
	// params
	w     *tensor.Tensor
	scale *tensor.Tensor
	// runtime
	freqs *tensor.Tensor
}

func NewAttention(dims, heads int, dropout float64, rope bool, opts ...LayerCreateOption) *Attention {
	var layer Attention
	layer.new("attention", opts...)
	layer.dims = dims
	layer.heads = heads
	layer.dropout = dropout
	layer.rope = rope
	if layer.dims%layer.heads != 0 {
		panic("dims must be divisible by heads")
	}
	layer.w = layer.initW(int64(dims*3), int64(dims*3))
	layer.scale = tensor.FromFloat32(nil, []float32{float32(math.Sqrt(float64(dims)))},
		tensor.WithShapes(1),
		tensor.WithDevice(layer.device))
	return &layer
}

func LoadAttention(name string, params map[string]*tensor.Tensor, args map[string]float32) Layer {
	var layer Attention
	layer.new("attention")
	layer.name = name
	layer.dims = int(args["dims"])
	layer.heads = int(args["heads"])
	layer.dropout = float64(args["dropout"])
	layer.rope = args["rope"] != 0
	layer.w = params["w"]
	layer.scale = tensor.FromFloat32(nil, []float32{float32(math.Sqrt(float64(layer.dims)))},
		tensor.WithShapes(1),
		tensor.WithDevice(layer.device))
	return &layer
}

func (layer *Attention) Forward(q, k, v, mask *tensor.Tensor, isCausal, train bool) *tensor.Tensor {
	if mask != nil && isCausal {
		panic("unexpected mask")
	}
	inputShape := q.Shapes()
	x := tensor.Cat([]*tensor.Tensor{q, k, v}, -1)           // (batch, seq, dims*3)
	x = x.MatMul(layer.w.Transpose(0, 1))                    // (batch, seq, dims*3)
	q = x.NArrow(-1, 0, int64(layer.dims))                   // (batch, seq, dims)
	k = x.NArrow(-1, int64(layer.dims), int64(layer.dims))   // (batch, seq, dims)
	v = x.NArrow(-1, int64(layer.dims*2), int64(layer.dims)) // (batch, seq, dims)
	q = layer.split(q)                                       // (batch, seq, heads, dims/heads)
	k = layer.split(k)                                       // (batch, seq, heads, dims/heads)
	v = layer.split(v)                                       // (batch, seq, heads, dims/heads)
	if layer.rope {
		q, k = layer.applyROPE(q, k, q.Shapes()[1])
	}
	q = q.Transpose(1, 2) // (batch, heads, seq, dims/heads)
	k = k.Transpose(1, 2) // (batch, heads, seq, dims/heads)
	v = v.Transpose(1, 2) // (batch, heads, seq, dims/heads)
	dropout := layer.dropout
	if !train {
		dropout = 0
	}
	y := tensor.ScaledDotProductAttention(q, k, v, mask, dropout, isCausal) // (batch, heads, seq, dims/heads)
	y = y.Transpose(1, 2)                                                   // (batch, seq, heads, dims/heads)
	y = y.Reshape(-1, inputShape[1], int64(layer.dims))                     // (batch, seq, dims)
	return y
}

func (layer *Attention) Score(q, k, v, mask *tensor.Tensor, isCausal, train bool) *tensor.Tensor {
	if mask != nil && isCausal {
		panic("unexpected mask")
	}
	x := tensor.Cat([]*tensor.Tensor{q, k, v}, -1)         // (batch, seq, dims*3)
	x = x.MatMul(layer.w)                                  // (batch, seq, dims*3)
	q = x.NArrow(-1, 0, int64(layer.dims))                 // (batch, seq, dims)
	k = x.NArrow(-1, int64(layer.dims), int64(layer.dims)) // (batch, seq, dims)
	q = layer.split(q)                                     // (batch, seq, heads, dims/heads)
	k = layer.split(k)                                     // (batch, seq, heads, dims/heads)
	if layer.rope {
		q, k = layer.applyROPE(q, k, q.Shapes()[1])
	}
	q = q.Transpose(1, 2) // (batch, heads, seq, dims/heads)
	k = k.Transpose(1, 2) // (batch, heads, seq, dims/heads)
	if isCausal {
		mask = buildCausal(q, k, layer.device)
	}
	score := q.MatMul(k.Transpose(-2, -1)).Div(layer.scale) // (batch, heads, seq, dims/heads)
	if mask != nil {
		score = score.Add(mask) // (batch, heads, seq, dims/heads)
	}
	return score.Softmax(-1) // (batch, heads, seq, dims/heads)
}

func buildFreqs(q *tensor.Tensor, dim, seq int64) *tensor.Tensor {
	data := make([]float32, dim/2)
	for i := int64(0); i < dim/2; i++ {
		data[i] = float32(1 / math.Pow(10000, float64(2*i)/float64(dim)))
	}
	freqs := tensor.FromFloat32(q.Storage(), data,
		tensor.WithShapes(dim/2),
		tensor.WithDevice(q.DeviceType()))
	data = make([]float32, seq)
	for i := range data {
		data[i] = float32(i)
	}
	t := tensor.FromFloat32(q.Storage(), data,
		tensor.WithShapes(seq),
		tensor.WithDevice(q.DeviceType()))
	freqs = tensor.Outer(t, freqs)
	data = make([]float32, freqs.ElemCount())
	for i := range data {
		data[i] = 1
	}
	ones := tensor.FromFloat32(q.Storage(), data,
		tensor.WithShapes(freqs.Shapes()...),
		tensor.WithDevice(q.DeviceType()))
	return tensor.Polar(ones, freqs).View(1, seq, 1, -1)
}

func (layer *Attention) applyROPE(q, k *tensor.Tensor, seq int64) (*tensor.Tensor, *tensor.Tensor) {
	qShapes := q.Shapes()
	kShapes := k.Shapes()
	xq := q.Reshape(append(qShapes[:len(qShapes)-1], -1, 2)...).ViewAsComplex()
	xk := k.Reshape(append(kShapes[:len(kShapes)-1], -1, 2)...).ViewAsComplex()
	if layer.freqs == nil || layer.freqs.Shapes()[1] < seq {
		layer.freqs = buildFreqs(q, qShapes[len(qShapes)-1], seq)
	}
	freqs := layer.freqs.NArrow(1, 0, seq)
	xq = xq.Mul(freqs).ViewAsReal().View(qShapes...)
	xk = xk.Mul(freqs).ViewAsReal().View(kShapes...)
	return xq, xk
}

func (layer *Attention) split(x *tensor.Tensor) *tensor.Tensor {
	return x.View(-1, x.Shapes()[1], int64(layer.heads), int64(layer.dims/layer.heads))
}

func (layer *Attention) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": layer.w,
	}
}

func (layer *Attention) Args() map[string]float32 {
	var rope float32
	if layer.rope {
		rope = 1
	}
	return map[string]float32{
		"dims":    float32(layer.dims),
		"heads":   float32(layer.heads),
		"dropout": float32(layer.dropout),
		"rope":    rope,
	}
}

func (layer *Attention) Freeze() {
	layer.w.SetRequiresGrad(false)
}

func (layer *Attention) Unfreeze() {
	layer.w.SetRequiresGrad(true)
}
