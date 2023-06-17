package model

import (
	"fmt"

	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
)

type transformer struct {
	attn   *layer.SelfAttention
	nor    *layer.Nor
	dense  *layer.Dense
	relu   *activation.ReLU
	output *layer.Dense
}

func newTransformer(i int) *transformer {
	attn := layer.NewSelfAttention(paddingSize, embeddingDim, heads)
	attn.SetName(fmt.Sprintf("transformer%d_attention", i))
	dense := layer.NewDense(embeddingDim * 4)
	dense.SetName(fmt.Sprintf("transformer%d_dense", i))
	output := layer.NewDense(embeddingDim)
	output.SetName(fmt.Sprintf("transformer%d_output", i))
	return &transformer{
		attn:   attn,
		nor:    layer.NewNor(),
		dense:  dense,
		relu:   activation.NewReLU(),
		output: output,
	}
}

var featureMask *tensor.Tensor

func init() {
	data := make([]float32, paddingSize*paddingSize)
	for i := 0; i < paddingSize; i++ {
		for j := i + 1; j < paddingSize; j++ {
			data[i*paddingSize+j] = -1e9
		}
	}
	featureMask = tensor.FromFloat32(nil, data, 1, 1, paddingSize, paddingSize)
}

func (t *transformer) forward(q, k *tensor.Tensor, padding []int, train bool) *tensor.Tensor {
	batchSize := q.Shapes()[0]
	paddingData := make([]float32, batchSize*maskSize)
	for i := 0; i < int(batchSize); i++ {
		start := i * maskSize
		for p := padding[i]; p < paddingSize; p++ {
			for j := 0; j < paddingSize; j++ {
				paddingData[start+p*paddingSize+j] = -1e9
				paddingData[start+j*paddingSize+p] = -1e9
			}
		}
	}
	paddingMask := tensor.FromFloat32(q.Storage(), paddingData, batchSize, 1, paddingSize, paddingSize)
	y := t.attn.Forward(q, k, paddingMask.Add(featureMask))
	y = y.Add(q)
	selfOut := t.nor.Forward(y)
	y = t.dense.Forward(y)
	y = t.relu.Forward(y)
	y = t.output.Forward(y)
	y = y.Add(selfOut)
	y = t.nor.Forward(y)
	return y
}

func (t *transformer) params() []*tensor.Tensor {
	var ret []*tensor.Tensor
	for _, p := range t.attn.Params() {
		ret = append(ret, p)
	}
	for _, p := range t.dense.Params() {
		ret = append(ret, p)
	}
	for _, p := range t.output.Params() {
		ret = append(ret, p)
	}
	return ret
}

func (t *transformer) layers() []layer.Layer {
	return []layer.Layer{
		t.attn,
		t.nor,
		t.dense,
		t.relu,
		t.output,
	}
}

func (t *transformer) loadFrom(layers []layer.Layer, idx int) int {
	t.attn = layers[idx].(*layer.SelfAttention)
	idx++
	t.nor = layers[idx].(*layer.Nor)
	idx++
	t.dense = layers[idx].(*layer.Dense)
	idx++
	t.relu = layers[idx].(*activation.ReLU)
	idx++
	t.output = layers[idx].(*layer.Dense)
	idx++
	return idx
}
