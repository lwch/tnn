package main

import (
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
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
	return &transformer{
		attn:    layer.NewSelfAttention(steps, dims),
		nor:     layer.NewNor(dims),
		flatten: layer.NewFlatten(),
		dense:   layer.NewDense(unitSize * 4),
		relu:    activation.NewReLU(),
		output:  layer.NewDense(unitSize),
	}
}

func (t *transformer) Forward(x *gorgonia.Node) (*gorgonia.Node, gorgonia.Nodes) {
	var params gorgonia.Nodes
	y, ps := t.attn.Forward(x)
	params = append(params, ps...)
	y = gorgonia.Must(gorgonia.Add(x, y))
	y = t.flatten.Forward(y)
	selfOut := t.nor.Forward(y)
	y, ps = t.dense.Forward(y)
	params = append(params, ps...)
	y = t.relu.Forward(y)
	y, ps = t.output.Forward(y)
	params = append(params, ps...)
	y = gorgonia.Must(gorgonia.Add(selfOut, y))
	y = t.nor.Forward(y)
	return gorgonia.Must(gorgonia.Reshape(y, tensor.Shape{batchSize, steps, dims})), params
}
