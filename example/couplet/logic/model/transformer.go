package model

import (
	"fmt"

	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
)

type transformer struct {
	attn   *layer.SelfAttention
	dense  *layer.Dense
	relu   *activation.ReLU
	norm1  *layer.LayerNorm
	norm2  *layer.LayerNorm
	output *layer.Dense
}

func newTransformer(i int) *transformer {
	attn := layer.NewSelfAttention(embeddingDim, heads, 0, false, layer.WithDevice(device))
	attn.SetName(fmt.Sprintf("transformer%d_attention", i))
	dense := layer.NewDense(embeddingDim*4, layer.WithDevice(device))
	dense.SetName(fmt.Sprintf("transformer%d_dense", i))
	output := layer.NewDense(embeddingDim, layer.WithDevice(device))
	output.SetName(fmt.Sprintf("transformer%d_output", i))
	norm1 := layer.NewLayerNorm(layer.WithDevice(device))
	norm1.SetName(fmt.Sprintf("transformer%d_norm1", i))
	norm2 := layer.NewLayerNorm(layer.WithDevice(device))
	norm2.SetName(fmt.Sprintf("transformer%d_norm2", i))
	return &transformer{
		attn:   attn,
		dense:  dense,
		relu:   activation.NewReLU(),
		norm1:  norm1,
		norm2:  norm2,
		output: output,
	}
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
	paddingMask := tensor.FromFloat32(q.Storage(), paddingData,
		tensor.WithShapes(batchSize, 1, paddingSize, paddingSize),
		tensor.WithDevice(device))
	y := t.attn.Forward(q, k, k, paddingMask, train)
	y = y.Add(q)
	selfOut := t.norm1.Forward(y)
	y = t.dense.Forward(y)
	y = t.relu.Forward(y)
	y = t.output.Forward(y)
	y = y.Add(selfOut)
	y = t.norm2.Forward(y)
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
	for _, p := range t.norm1.Params() {
		ret = append(ret, p)
	}
	for _, p := range t.norm2.Params() {
		ret = append(ret, p)
	}
	return ret
}

func (t *transformer) layers() []layer.Layer {
	return []layer.Layer{
		t.attn,
		t.dense,
		t.relu,
		t.norm1,
		t.norm2,
		t.output,
	}
}

func (t *transformer) loadFrom(layers []layer.Layer, idx int) int {
	t.attn = layers[idx].(*layer.SelfAttention)
	idx++
	t.dense = layers[idx].(*layer.Dense)
	idx++
	t.relu = layers[idx].(*activation.ReLU)
	idx++
	t.norm1 = layers[idx].(*layer.LayerNorm)
	idx++
	t.norm2 = layers[idx].(*layer.LayerNorm)
	idx++
	t.output = layers[idx].(*layer.Dense)
	idx++
	return idx
}
