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
	rope        bool
	ropeBase    int64
	// params
	q, k, v *tensor.Tensor
	scale   *tensor.Tensor
	// runtime
	freqs *tensor.Tensor
}

func NewAttention1(name string, dims, heads int, dropout float64, rope bool, opts ...LayerCreateOption) *Attention1 {
	var layer Attention1
	layer.new("attention1", name, opts...)
	layer.dims = dims
	layer.heads = heads
	layer.dropout = dropout
	layer.rope = rope
	layer.ropeBase = 10000
	if layer.dims%layer.heads != 0 {
		panic("dims must be divisible by heads")
	}
	layer.q = layer.initW(int64(dims), int64(dims))
	layer.k = layer.initW(int64(dims), int64(dims))
	layer.v = layer.initW(int64(dims), int64(dims))
	layer.scale = layer.initN(math.Sqrt(float64(dims)))
	return &layer
}

func LoadAttention1(name string, params []*tensor.Tensor, args map[string]float32) Layer {
	var layer Attention1
	layer.new("attention1", name)
	layer.dims = int(args["dims"])
	layer.heads = int(args["heads"])
	layer.dropout = float64(args["dropout"])
	layer.rope = args["rope"] != 0
	layer.ropeBase = int64(args["rope_base"])
	if layer.ropeBase <= 0 {
		layer.ropeBase = 10000
	}
	layer.q = params[0]
	layer.k = params[1]
	layer.v = params[2]
	layer.scale = layer.initN(math.Sqrt(float64(layer.dims)))
	return &layer
}

func (layer *Attention1) SetROPEBase(n int64) {
	layer.ropeBase = n
}

func (layer *Attention1) Forward(q, k, v, mask *tensor.Tensor, isCausal, train bool) *tensor.Tensor {
	if mask != nil && isCausal {
		panic("unexpected mask")
	}
	inputShape := q.Shapes()
	q = q.MatMul(layer.q.Transpose(0, 1)) // (batch, seq, dims)
	k = k.MatMul(layer.k.Transpose(0, 1)) // (batch, seq, dims)
	v = v.MatMul(layer.v.Transpose(0, 1)) // (batch, seq, dims)
	q = layer.split(q)                    // (batch, heads, seq, dims/heads)
	k = layer.split(k)                    // (batch, heads, seq, dims/heads)
	v = layer.split(v)                    // (batch, heads, seq, dims/heads)
	if layer.rope {
		q, k = layer.applyROPE(q, k, q.Shapes()[1])
	}
	q = q.Transpose(1, 2) // (batch, heads, seq, dims/heads)
	k = k.Transpose(1, 2) // (batch, heads, seq, dims/heads)
	v = v.Transpose(1, 2) // (batch, heads, seq, dims/heads)
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
	q = q.MatMul(layer.q.Transpose(0, 1)) // (batch, seq, dims)
	k = k.MatMul(layer.k.Transpose(0, 1)) // (batch, seq, dims)
	q = layer.split(q)                    // (batch, heads, seq, dims/heads)
	k = layer.split(k)                    // (batch, heads, seq, dims/heads)
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
	return score.Softmax1(-1) // (batch, heads, seq, dims/heads)
}

func (layer *Attention1) applyROPE(q, k *tensor.Tensor, seq int64) (*tensor.Tensor, *tensor.Tensor) {
	qShapes := q.Shapes()
	kShapes := k.Shapes()
	xq := q.Reshape(append(qShapes[:len(qShapes)-1], -1, 2)...).
		ToDevice(consts.KCPU).ToScalarType(consts.KFloat).
		ViewAsComplex()
	xk := k.Reshape(append(kShapes[:len(kShapes)-1], -1, 2)...).
		ToDevice(consts.KCPU).ToScalarType(consts.KFloat).
		ViewAsComplex()
	if layer.freqs == nil || layer.freqs.Shapes()[1] < seq {
		layer.freqs = buildFreqs(q.DeviceType(), layer.ropeBase, qShapes[len(qShapes)-1], seq)
	}
	freqs := layer.freqs.NArrow(1, 0, seq).ToDevice(consts.KCPU)
	xq = xq.Mul(freqs).ViewAsReal().Flatten(3, -1).
		ToDevice(q.DeviceType()).ToScalarType(q.ScalarType())
	xk = xk.Mul(freqs).ViewAsReal().Flatten(3, -1).
		ToDevice(k.DeviceType()).ToScalarType(k.ScalarType())
	return xq, xk
}

func (layer *Attention1) split(x *tensor.Tensor) *tensor.Tensor {
	return x.View(-1, x.Shapes()[1], int64(layer.heads), int64(layer.dims/layer.heads))
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
	return tensor.FromFloat32(mask,
		tensor.WithShapes(1, 1, l, s),
		tensor.WithDevice(device))
}

func (layer *Attention1) Params() []*tensor.Tensor {
	return []*tensor.Tensor{
		layer.q,
		layer.k,
		layer.v,
	}
}

func (layer *Attention1) Args() map[string]float32 {
	var rope float32
	if layer.rope {
		rope = 1
	}
	return map[string]float32{
		"dims":      float32(layer.dims),
		"heads":     float32(layer.heads),
		"dropout":   float32(layer.dropout),
		"rope":      rope,
		"rope_base": float32(layer.ropeBase),
	}
}

func (layer *Attention1) Freeze() {
	layer.q.SetRequiresGrad(false)
	layer.k.SetRequiresGrad(false)
	layer.v.SetRequiresGrad(false)
}

func (layer *Attention1) Unfreeze() {
	layer.q.SetRequiresGrad(true)
	layer.k.SetRequiresGrad(true)
	layer.v.SetRequiresGrad(true)
}

func (layer *Attention1) ToScalarType(t consts.ScalarType) {
	layer.q = layer.q.ToScalarType(t)
	layer.k = layer.k.ToScalarType(t)
	layer.v = layer.v.ToScalarType(t)
}

func (layer *Attention1) Reset() {
	layer.q = layer.initW(layer.q.Shapes()...)
	layer.k = layer.initW(layer.k.Shapes()...)
	layer.v = layer.initW(layer.v.Shapes()...)
}

func (layer *Attention1) Clone() Layer {
	return &Attention1{
		base:    layer.base.clone(),
		dims:    layer.dims,
		heads:   layer.heads,
		dropout: layer.dropout,
		rope:    layer.rope,
		q:       layer.q.Clone(),
		k:       layer.k.Clone(),
		v:       layer.v.Clone(),
		scale:   layer.scale.Clone(),
	}
}
