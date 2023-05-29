package layer

import (
	"fmt"
	"math"

	m "github.com/lwch/tnn/internal/math"
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
	size := dims / headSize * seqSize
	for i := 0; i < headSize; i++ {
		suffix := fmt.Sprintf("H%d", i)
		shapes["Wq"+suffix] = Shape{size, size}
		shapes["Bq"+suffix] = Shape{1, size}
		shapes["Wk"+suffix] = Shape{size, size}
		shapes["Bk"+suffix] = Shape{1, size}
		shapes["Wv"+suffix] = Shape{size, size}
		shapes["Bv"+suffix] = Shape{1, size}
	}
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
	return layer.ForwardQKV(input, input, input, false, isTraining)
}

func (layer *SelfAttention) ForwardQKV(q, k, v *tensor.Tensor, mask, isTraining bool) *tensor.Tensor {
	if !layer.hasInit {
		layer.initParams()
	}
	if layer.headSize == 1 {
		return layer.forwardSingleHead(q, k, v, mask)
	}
	batchSize, _ := q.Dims()
	var ret *tensor.Tensor
	sizePreHead := layer.dims / layer.headSize
	for head := 0; head < layer.headSize; head++ {
		var inputQ, inputK, inputV *tensor.Tensor
		for seq := 0; seq < layer.seqSize; seq++ {
			start := seq * layer.dims
			start += head * sizePreHead
			qHead := q.Slice(0, batchSize, start, start+sizePreHead)
			kHead := k.Slice(0, batchSize, start, start+sizePreHead)
			vHead := v.Slice(0, batchSize, start, start+sizePreHead)
			if inputQ == nil {
				inputQ = qHead
				inputK = kHead
				inputV = vHead
			} else {
				inputQ = inputQ.Stack(qHead)
				inputK = inputK.Stack(kHead)
				inputV = inputV.Stack(vHead)
			}
		}
		suffix := fmt.Sprintf("H%d", head)
		Wq := layer.params.Get("Wq" + suffix)
		Wk := layer.params.Get("Wk" + suffix)
		Wv := layer.params.Get("Wv" + suffix)
		Bq := layer.params.Get("Bq" + suffix)
		Bk := layer.params.Get("Bk" + suffix)
		Bv := layer.params.Get("Bv" + suffix)
		dq := inputQ.Mul(Wq).Add(Bq)
		dk := inputK.Mul(Wk).Add(Bk)
		dv := inputV.Mul(Wv).Add(Bv)
		a := dq.Mul(dk.T())
		a = a.Scale(layer.scale)
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
		a = a.Mul(dv)
		if ret == nil {
			ret = a
		} else {
			ret = ret.Stack(a)
		}
	}
	return ret
}

func (layer *SelfAttention) forwardSingleHead(q, k, v *tensor.Tensor, mask bool) *tensor.Tensor {
	Wq := layer.params.Get("WqH0")
	Wk := layer.params.Get("WkH0")
	Wv := layer.params.Get("WvH0")
	Bq := layer.params.Get("BqH0")
	Bk := layer.params.Get("BkH0")
	Bv := layer.params.Get("BvH0")
	dq := q.Mul(Wq).Add(Bq)
	dk := k.Mul(Wk).Add(Bk)
	dv := v.Mul(Wv).Add(Bv)
	a := dq.Mul(dk.T())
	a = a.Scale(layer.scale)
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
	return a.Mul(dv)
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
