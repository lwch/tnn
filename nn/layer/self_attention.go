package layer

import (
	"fmt"
	"math"

	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
)

type SelfAttention struct {
	*base
	seqSize  int
	dims     int
	headSize int
	scale    float64
}

func NewSelfAttention(seqSize, dims, headSize int, init initializer.Initializer) Layer {
	var layer SelfAttention
	shapes := make(map[string]Shape)
	for i := 0; i < headSize; i++ {
		suffix := fmt.Sprintf("H%d", i)
		shapes["Wq"+suffix] = Shape{seqSize, seqSize}
		shapes["Wk"+suffix] = Shape{seqSize, seqSize}
		shapes["Wv"+suffix] = Shape{seqSize, seqSize}
		shapes["Bq"+suffix] = Shape{seqSize, 1}
		shapes["Bk"+suffix] = Shape{seqSize, 1}
		shapes["Bv"+suffix] = Shape{seqSize, 1}
	}
	shapes["Wo"] = Shape{dims * headSize, dims}
	layer.base = new("self_attention", shapes, init)
	layer.seqSize = seqSize
	layer.dims = dims
	layer.headSize = headSize
	layer.scale = 1 / math.Sqrt(float64(layer.dims))
	return &layer
}

func LoadSelfAttention(name string, params map[string]*pb.Dense, args map[string]*pb.Dense) Layer {
	var layer SelfAttention
	layer.base = new("self_attention", nil, nil)
	p := args["params"].GetData()
	layer.seqSize = int(p[0])
	layer.dims = int(p[1])
	layer.headSize = int(p[2])
	layer.scale = 1 / math.Sqrt(float64(layer.dims))
	layer.name = name
	layer.base.loadParams(params)
	return &layer
}

func (layer *SelfAttention) Forward(input *tensor.Tensor, isTraining bool) *tensor.Tensor {
	return layer.ForwardQKV(input, input, input, nil, isTraining)
}

func (layer *SelfAttention) ForwardQKV(q, k, v *tensor.Tensor, masks []*tensor.Tensor, isTraining bool) *tensor.Tensor {
	if !layer.hasInit {
		layer.initParams()
	}
	if layer.headSize == 1 {
		return layer.forwardSingleHead(q, k, v, masks)
	}
	batchSize, _ := q.Dims()
	var ret *tensor.Tensor
	Wo := layer.params.Get("Wo")
	for batch := 0; batch < batchSize; batch++ {
		var row *tensor.Tensor
		for head := 0; head < layer.headSize; head++ {
			inputQ := q.Row2Matrix(batch, layer.seqSize, layer.dims)
			inputK := k.Row2Matrix(batch, layer.seqSize, layer.dims)
			inputV := v.Row2Matrix(batch, layer.seqSize, layer.dims)
			suffix := fmt.Sprintf("H%d", head)
			Wq := layer.params.Get("Wq" + suffix)
			Wk := layer.params.Get("Wk" + suffix)
			Wv := layer.params.Get("Wv" + suffix)
			Bq := layer.params.Get("Bq" + suffix)
			Bk := layer.params.Get("Bk" + suffix)
			Bv := layer.params.Get("Bv" + suffix)
			dq := Wq.Mul(inputQ).Add(Bq) // (seq, dims)
			dk := Wk.Mul(inputK).Add(Bk) // (seq, dims)
			dv := Wv.Mul(inputV).Add(Bv) // (seq, dims)
			a := dq.Mul(dk.T())          // (seq, seq)
			a = a.Scale(layer.scale)     // (seq, seq)
			if batch < len(masks) {
				a = a.Add(masks[batch]) // (seq, seq)
			}
			a = a.Softmax(1) // (seq, seq)
			a = a.Mul(dv)    // (seq, dims)
			if row == nil {
				row = a
			} else {
				row = row.Conact(a)
			}
		}
		row = row.Mul(Wo).RowVector()
		if ret == nil {
			ret = row
		} else {
			ret = ret.Stack(row)
		}
	}
	return ret
}

func (layer *SelfAttention) forwardSingleHead(q, k, v *tensor.Tensor, masks []*tensor.Tensor) *tensor.Tensor {
	batchSize, _ := q.Dims()
	Wq := layer.params.Get("WqH0")
	Wk := layer.params.Get("WkH0")
	Wv := layer.params.Get("WvH0")
	Bq := layer.params.Get("BqH0")
	Bk := layer.params.Get("BkH0")
	Bv := layer.params.Get("BvH0")
	var ret *tensor.Tensor
	for batch := 0; batch < batchSize; batch++ {
		inputQ := q.Row2Matrix(batch, layer.seqSize, layer.dims)
		inputK := k.Row2Matrix(batch, layer.seqSize, layer.dims)
		inputV := v.Row2Matrix(batch, layer.seqSize, layer.dims)
		dq := Wq.Mul(inputQ).Add(Bq)
		dk := Wk.Mul(inputK).Add(Bk)
		dv := Wv.Mul(inputV).Add(Bv)
		a := dq.Mul(dk.T())
		a = a.Scale(layer.scale)
		if batch < len(masks) {
			a = a.Add(masks[batch])
		}
		a = a.Softmax(1)
		row := a.Mul(dv).RowVector()
		if ret == nil {
			ret = row
		} else {
			ret = ret.Stack(row)
		}
	}
	return ret
}

func (layer *SelfAttention) Args() map[string]*mat.VecDense {
	return map[string]*mat.VecDense{
		"params": mat.NewVecDense(3, []float64{
			float64(layer.seqSize),
			float64(layer.dims),
			float64(layer.headSize),
		}),
	}
}
