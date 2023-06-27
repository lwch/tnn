package main

import (
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
)

type transformer struct {
	attn    *layer.SelfAttention
	nor     *layer.Nor
	flatten *layer.Flatten
	dense   *layer.Dense
	sigmoid *activation.Sigmoid
	output  *layer.Dense
}

func newTransformer() *transformer {
	return &transformer{
		attn:    layer.NewSelfAttention(dims, 1, 0.1, device),
		nor:     layer.NewNor(device),
		flatten: layer.NewFlatten(),
		dense:   layer.NewDense(unitSize*4, device),
		sigmoid: activation.NewSigmoid(),
		output:  layer.NewDense(unitSize, device),
	}
}

func (t *transformer) Forward(x *tensor.Tensor) *tensor.Tensor {
	y := t.attn.Forward(x, x, x, nil, true)
	y = y.Add(x)
	selfOut := t.nor.Forward(y)
	y = t.flatten.Forward(y)
	y = t.dense.Forward(y)
	y = t.sigmoid.Forward(y)
	y = t.output.Forward(y)
	y = y.Reshape(batchSize, steps, dims)
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
