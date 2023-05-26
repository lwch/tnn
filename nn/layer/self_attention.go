package layer

import (
	"math"

	m "github.com/lwch/tnn/internal/math"
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
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
	Wk := layer.params.Get("Wk")
	Wv := layer.params.Get("Wv")
	Bq := layer.params.Get("Bq")
	Bk := layer.params.Get("Bk")
	Bv := layer.params.Get("Bv")
	q := input.Mul(Wq).AddVector(Bq)
	k := input.Mul(Wk).AddVector(Bk)
	v := input.Mul(Wv).AddVector(Bv)
	a := k.T().Mul(q)
	hasNan(a.Value())
	a = a.Scale(1 / math.Sqrt(float64(layer.dims)))
	a = m.Softmax(a, 0)
	return v.Mul(a)
}

func hasNan(x *mat.Dense) {
	rows, cols := x.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if math.IsNaN(x.At(i, j)) {
				panic("nan")
			}
		}
	}
}

func (layer *SelfAttention) ForwardQKV(q, k, v *tensor.Tensor, isTraining bool) *tensor.Tensor {
	if !layer.hasInit {
		layer.initParams()
	}
	Wq := layer.params.Get("Wq")
	Wk := layer.params.Get("Wk")
	Wv := layer.params.Get("Wv")
	Bq := layer.params.Get("Bq")
	Bk := layer.params.Get("Bk")
	Bv := layer.params.Get("Bv")
	q = q.Mul(Wq).AddVector(Bq)
	k = k.Mul(Wk).AddVector(Bk)
	v = v.Mul(Wv).AddVector(Bv)
	a := k.T().Mul(q)
	a = a.Scale(1 / math.Sqrt(float64(layer.dims)))
	a = m.Softmax(a, 0)
	return v.Mul(a)
}

func (layer *SelfAttention) Args() map[string]*mat.VecDense {
	return map[string]*mat.VecDense{
		"dims": mat.NewVecDense(1, []float64{float64(layer.dims)}),
	}
}
