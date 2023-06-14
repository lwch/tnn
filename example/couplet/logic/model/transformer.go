package model

import (
	"fmt"

	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
)

type transformer struct {
	attn    *layer.SelfAttention
	nor     *layer.Nor
	flatten *layer.Flatten
	dense   *layer.Dense
	relu    *activation.ReLU
	output  *layer.Dense
}

func newTransformer(i int) *transformer {
	attn := layer.NewSelfAttention(paddingSize, embeddingDim, heads)
	attn.SetName(fmt.Sprintf("transformer%d_attention", i))
	dense := layer.NewDense(unitSize * 4)
	dense.SetName(fmt.Sprintf("transformer%d_dense", i))
	output := layer.NewDense(unitSize)
	output.SetName(fmt.Sprintf("transformer%d_output", i))
	return &transformer{
		attn:    attn,
		nor:     layer.NewNor(),
		flatten: layer.NewFlatten(),
		dense:   dense,
		relu:    activation.NewReLU(),
		output:  output,
	}
}

func (t *transformer) forward(q, k *tensor.Tensor, train bool) *tensor.Tensor {
	batchSize := q.Shapes()[0]
	y := t.attn.Forward(q, k)
	y = y.Add(q)
	selfOut := t.nor.Forward(y)
	y = t.flatten.Forward(y)
	y = t.dense.Forward(y)
	y = t.relu.Forward(y)
	y = t.output.Forward(y)
	y = y.Reshape(batchSize, paddingSize, embeddingDim)
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
		t.flatten,
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
	t.flatten = layers[idx].(*layer.Flatten)
	idx++
	t.dense = layers[idx].(*layer.Dense)
	idx++
	t.relu = layers[idx].(*activation.ReLU)
	idx++
	t.output = layers[idx].(*layer.Dense)
	idx++
	return idx
}
