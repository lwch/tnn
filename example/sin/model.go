package main

import (
	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/optimizer"
	"gorgonia.org/gorgonia"
)

type model struct {
	rnn         *layer.Rnn
	lstm        *layer.Lstm
	flatten     *layer.Flatten
	outputLayer *layer.Dense
	hidden      *gorgonia.Node
	cell        *gorgonia.Node
}

func newModel() *model {
	return &model{
		rnn: layer.NewRnn(featureSize, steps, hiddenSize),
		// lstm:        layer.NewLstm(featureSize, steps, hiddenSize),
		flatten:     layer.NewFlatten(),
		outputLayer: layer.NewDense(1),
	}
}

func (m *model) Forward(loss loss.Loss, x, y *gorgonia.Node) (gorgonia.VM, *gorgonia.Node, *gorgonia.Node) {
	var output *gorgonia.Node
	if m.rnn != nil {
		var err error
		output, m.hidden, err = m.rnn.Forward(x, m.hidden)
		runtime.Assert(err)
	} else {
		var err error
		output, m.hidden, m.cell, err = m.lstm.Forward(x, m.hidden, m.cell)
		runtime.Assert(err)
	}
	output = m.flatten.Forward(output)
	output = m.outputLayer.Forward(output)
	var lossValue *gorgonia.Node
	if loss != nil {
		lossValue = loss.Loss(y, output)
		_, err := gorgonia.Grad(lossValue, m.Params()...)
		runtime.Assert(err)
	}
	return gorgonia.NewTapeMachine(x.Graph(),
			gorgonia.BindDualValues(m.Params()...)),
		output, lossValue
}

func (m *model) Train(epoch int, loss loss.Loss, optimizer optimizer.Optimizer, x, y *gorgonia.Node) {
	if epoch%clearSteps == 0 {
		m.hidden = nil
		m.cell = nil
	}
	vm, _, _ := m.Forward(loss, x, y)
	defer vm.Close()
	runtime.Assert(vm.RunAll())
	runtime.Assert(optimizer.Step(m.Params()))
}

func (m *model) Predict(x *gorgonia.Node) gorgonia.Value {
	vm, pred, _ := m.Forward(nil, x, nil)
	defer vm.Close()
	runtime.Assert(vm.RunAll())
	return pred.Value()
}

func (m *model) Loss(loss loss.Loss, x, y *gorgonia.Node) float32 {
	vm, _, lossValue := m.Forward(loss, x, y)
	defer vm.Close()
	runtime.Assert(vm.RunAll())
	return lossValue.Value().Data().(float32)
}

func (m *model) Params() gorgonia.Nodes {
	var ret gorgonia.Nodes
	if m.rnn != nil {
		ret = append(ret, m.rnn.Params()...)
	}
	if m.lstm != nil {
		ret = append(ret, m.lstm.Params()...)
	}
	ret = append(ret, m.outputLayer.Params()...)
	return ret
}
