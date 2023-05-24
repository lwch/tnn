package layer

import (
	"math"

	m "github.com/lwch/tnn/internal/math"
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/tensor"
)

type SelfAttention struct {
	*base
	dims int
}

func NewSelfAttention(dims int, init initializer.Initializer) Layer {
	var layer SelfAttention
	layer.base = new("self_attention", map[string]Shape{
		"Wq": {dims, dims},
		"Bq": {1, dims},
		"Wk": {dims, dims},
		"Bk": {1, dims},
		"Wv": {dims, dims},
		"Bv": {1, dims},
	}, init)
	layer.dims = dims
	return &layer
}

func LoadSelfAttention(name string, params map[string]*pb.Dense, args map[string]*pb.Dense) Layer {
	var layer SelfAttention
	layer.base = new("self_attention", nil, nil)
	layer.dims = int(args["dims"].GetData()[0])
	layer.name = name
	layer.base.loadParams(params)
	return &layer
}

func (layer *SelfAttention) Forward(input *tensor.Tensor, isTraining bool) *tensor.Tensor {
	if !layer.hasInit {
		layer.initParams()
	}
	Wq := layer.params.Get("Wq")
	Bq := layer.params.Get("Bq")
	Wk := layer.params.Get("Wk")
	Bk := layer.params.Get("Bk")
	Wv := layer.params.Get("Wv")
	Bv := layer.params.Get("Bv")
	q := input.Mul(Wq).AddVector(Bq)
	k := input.Mul(Wk).AddVector(Bk)
	v := input.Mul(Wv).AddVector(Bv)
	a := k.T().Mul(q)
	a = a.Scale(1 / math.Sqrt(float64(layer.dims)))
	a = m.Softmax(a, 0)
	return v.Mul(a)
}
