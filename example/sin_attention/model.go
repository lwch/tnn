package main

import (
	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/optimizer"
	"gorgonia.org/gorgonia"
)

type model struct {
	attn        []*transformer
	flatten     *layer.Flatten
	sigmoid     *activation.Sigmoid
	outputLayer *layer.Dense
	vm          gorgonia.VM
	pred        *gorgonia.Node
	loss        *gorgonia.Node
}

func newModel(g *gorgonia.ExprGraph) *model {
	var m model
	for i := 0; i < transformerSize; i++ {
		m.attn = append(m.attn, newTransformer())
	}
	m.flatten = layer.NewFlatten()
	m.sigmoid = activation.NewSigmoid()
	m.outputLayer = layer.NewDense(1)
	return &m
}

func (m *model) Compile(loss loss.Loss, x, y *gorgonia.Node) {
	output := x
	for _, attn := range m.attn {
		output = attn.Forward(output)
	}
	output = m.flatten.Forward(output)
	output = m.sigmoid.Forward(output)
	output = m.outputLayer.Forward(output)
	var lossValue *gorgonia.Node
	lossValue = loss.Loss(y, output)
	_, err := gorgonia.Grad(lossValue, m.Params()...)
	runtime.Assert(err)
	m.vm = gorgonia.NewTapeMachine(x.Graph(),
		gorgonia.BindDualValues(m.Params()...))
	m.pred = output
	m.loss = lossValue
}

func (m *model) Train(optimizer optimizer.Optimizer, x, y *gorgonia.Node) {
	m.vm.Reset()
	runtime.Assert(m.vm.RunAll())
	runtime.Assert(optimizer.Step(m.Params()))
}

func (m *model) Predict(x *gorgonia.Node) gorgonia.Value {
	m.vm.Reset()
	runtime.Assert(m.vm.RunAll())
	return m.pred.Value()
}

func (m *model) Loss(x *gorgonia.Node) float32 {
	m.vm.Reset()
	runtime.Assert(m.vm.RunAll())
	return m.loss.Value().Data().(float32)
}

func (m *model) Params() gorgonia.Nodes {
	var ret gorgonia.Nodes
	for _, attn := range m.attn {
		ret = append(ret, attn.Params()...)
	}
	ret = append(ret, m.outputLayer.Params()...)
	return ret
}
