package main

import (
	"github.com/lwch/gotorch/optimizer"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
)

type model struct {
	attn        []*transformer
	flatten     *layer.Flatten
	sigmoid     *activation.Sigmoid
	outputLayer *layer.Dense
	optimizer   optimizer.Optimizer
}

func newModel(optimizer optimizer.Optimizer) *model {
	var m model
	for i := 0; i < transformerSize; i++ {
		m.attn = append(m.attn, newTransformer())
	}
	m.flatten = layer.NewFlatten()
	m.sigmoid = activation.NewSigmoid()
	m.outputLayer = layer.NewDense(1)
	m.optimizer = optimizer
	return &m
}

func (m *model) Forward(x *tensor.Tensor) *tensor.Tensor {
	y := x
	for _, attn := range m.attn {
		y = attn.Forward(y)
	}
	y = m.flatten.Forward(y)
	y = m.sigmoid.Forward(y)
	y = m.outputLayer.Forward(y)
	return y
}

func (m *model) Train(x, y *tensor.Tensor) {
	pred := m.Forward(x)
	l := lossFunc(pred, y)
	l.Backward()
}

func (m *model) Apply() {
	m.optimizer.Step(m.params())
}

func (m *model) Predict(x *tensor.Tensor) []float32 {
	return m.Forward(x).Float32Value()
}

func (m *model) Loss(x, y *tensor.Tensor) float32 {
	pred := m.Forward(x)
	return float32(lossFunc(pred, y).Value())
}

func (m *model) params() []*tensor.Tensor {
	var ret []*tensor.Tensor
	for _, attn := range m.attn {
		ret = append(ret, attn.params()...)
	}
	for _, p := range m.outputLayer.Params() {
		ret = append(ret, p)
	}
	return ret
}
