package main

import (
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
)

type transformer struct {
	attn    *layer.Attention
	dense   *layer.Linear
	sigmoid *activation.Sigmoid
	norm1   *layer.LayerNorm
	norm2   *layer.LayerNorm
	output  *layer.Linear
}

func newTransformer() *transformer {
	return &transformer{
		attn:    layer.NewAttention("attn", dims, 1, 0.1, false, layer.WithDevice(device)),
		dense:   layer.NewLinear("attn.l1", dims, dims*4, layer.WithDevice(device)),
		sigmoid: activation.NewSigmoid(),
		norm1:   layer.NewLayerNorm("attn.norm1", dims, layer.WithDevice(device)),
		norm2:   layer.NewLayerNorm("attn.norm2", dims, layer.WithDevice(device)),
		output:  layer.NewLinear("attn.l2", dims*4, dims, layer.WithDevice(device)),
	}
}

func (t *transformer) Forward(x *tensor.Tensor, train bool) *tensor.Tensor {
	y := t.attn.Forward(x, x, x, nil, true, train)
	y = y.Add(x)
	selfOut := t.norm1.Forward(y)
	y = t.dense.Forward(y)
	y = t.sigmoid.Forward(y)
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
