package model

import (
	"fmt"

	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
)

type transformer struct {
	attn   *layer.Attention
	dense  *layer.Linear
	relu   *activation.ReLU
	norm1  *layer.LayerNorm
	norm2  *layer.LayerNorm
	output *layer.Linear
}

func newTransformer(i int) *transformer {
	attn := layer.NewAttention(fmt.Sprintf("attn.%d", i), embeddingDim, heads, 0, false, layer.WithDevice(device))
	dense := layer.NewLinear(fmt.Sprintf("attn.%d.l1", i), embeddingDim, embeddingDim*4, layer.WithDevice(device))
	output := layer.NewLinear(fmt.Sprintf("attn.%d.output", i), embeddingDim*4, embeddingDim, layer.WithDevice(device))
	norm1 := layer.NewLayerNorm(fmt.Sprintf("attn.%d.norm1", i), embeddingDim, layer.WithDevice(device))
	norm2 := layer.NewLayerNorm(fmt.Sprintf("attn.%d.norm2", i), embeddingDim, layer.WithDevice(device))
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
	maskData := make([]float32, batchSize*maskSize)
	// padding mask
	for i := 0; i < int(batchSize); i++ {
		start := i * maskSize
		for p := padding[i]; p < paddingSize; p++ {
			for j := 0; j < paddingSize; j++ {
				maskData[start+p*paddingSize+j] = -1e9
				maskData[start+j*paddingSize+p] = -1e9
			}
		}
	}
	for y := 0; y < paddingSize; y++ {
		for x := 0; x < paddingSize; x++ {
			if x > y {
				maskData[y*paddingSize+x] = -1e9
			}
		}
	}
	mask := tensor.FromFloat32(maskData,
		tensor.WithShapes(batchSize, 1, paddingSize, paddingSize),
		tensor.WithDevice(device))
	y := t.attn.Forward(q, k, k, mask, false, train)
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
	t.attn = layers[idx].(*layer.Attention)
	idx++
	t.dense = layers[idx].(*layer.Linear)
	idx++
	t.relu = layers[idx].(*activation.ReLU)
	idx++
	t.norm1 = layers[idx].(*layer.LayerNorm)
	idx++
	t.norm2 = layers[idx].(*layer.LayerNorm)
	idx++
	t.output = layers[idx].(*layer.Linear)
	idx++
	return idx
}
