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
	p := args["params"].GetData()
	layer.dims = int(p[0])
	layer.name = name
	layer.base.loadParams(params)
	return &layer
}

func (layer *SelfAttention) Forward(input *tensor.Tensor, isTraining bool) *tensor.Tensor {
	return layer.ForwardQKV(input, input, input, false, isTraining)
}

func (layer *SelfAttention) ForwardQKV(q, k, v *tensor.Tensor, mask, isTraining bool) *tensor.Tensor {
	if !layer.hasInit {
		layer.initParams()
	}
	// batchSize, _ := q.Dims()
	// var ret *tensor.Tensor
	// for i := 0; i < 1; i++ {
	// 	start := i * layer.dims
	// 	q := q.Slice(0, batchSize, start, start+layer.dims)
	// 	k := k.Slice(0, batchSize, start, start+layer.dims)
	// 	v := v.Slice(0, batchSize, start, start+layer.dims)
	Wq := layer.params.Get("Wq")
	Wk := layer.params.Get("Wk")
	Wv := layer.params.Get("Wv")
	Bq := layer.params.Get("Bq")
	Bk := layer.params.Get("Bk")
	Bv := layer.params.Get("Bv")
	q = q.Mul(Wq).Add(Bq)
	k = k.Mul(Wk).Add(Bk)
	v = v.Mul(Wv).Add(Bv)
	a := q.Mul(k.T())
	a = a.Scale(1 / math.Sqrt(float64(layer.dims)))
	if mask {
		rows, cols := a.Dims()
		data := make([]float64, rows*cols)
		for i := 0; i < rows; i++ {
			for j := 0; j <= i; j++ {
				data[i*cols+j] = 1
			}
			for j := i + 1; j < cols; j++ {
				data[i*cols+j] = -1e9
			}
		}
		a = a.MulElem(tensor.New(data, rows, cols))
	}
	a = m.Softmax(a, 1)
	a = a.Mul(v)
	return a
	// 	if ret == nil {
	// 		ret = a
	// 	} else {
	// 		ret = ret.Stack(a)
	// 	}
	// }
	// return ret
}

func (layer *SelfAttention) Args() map[string]*mat.VecDense {
	return map[string]*mat.VecDense{
		"params": mat.NewVecDense(1, []float64{
			float64(layer.dims),
		}),
	}
}
