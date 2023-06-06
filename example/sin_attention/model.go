package main

import (
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
)

type model struct {
	attn        []*transformer
	flatten     *layer.Flatten
	sigmoid     *activation.Sigmoid
	outputLayer *layer.Dense
}

func newModel() *model {
	var m model
	for i := 0; i < transformerSize; i++ {
		m.attn = append(m.attn, newTransformer(i))
	}
	m.flatten = layer.NewFlatten()
	m.sigmoid = activation.NewSigmoid()
	m.outputLayer = layer.NewDense(1)
	return &m
}
