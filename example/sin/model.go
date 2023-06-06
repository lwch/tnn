package main

import (
	"sort"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/optimizer"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type model struct {
	g           *gorgonia.ExprGraph
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

func (m *model) Forward(loss loss.Loss, x, y tensor.Tensor) (gorgonia.VM, *gorgonia.Node, *gorgonia.Node, gorgonia.Nodes) {
	xs := gorgonia.NodeFromAny(m.g, x, gorgonia.WithName("x"))
	var output *gorgonia.Node
	if m.rnn != nil {
		var err error
		output, m.hidden, err = m.rnn.Forward(xs, m.hidden)
		runtime.Assert(err)
	} else {
		var err error
		output, m.hidden, m.cell, err = m.lstm.Forward(xs, m.hidden, m.cell)
		runtime.Assert(err)
	}
	output = m.flatten.Forward(output)
	output = m.outputLayer.Forward(output)
	var lossValue *gorgonia.Node
	params := m.Params(m.g)
	if loss != nil {
		ys := gorgonia.NodeFromAny(m.g, y, gorgonia.WithName("y"))
		lossValue = loss.Loss(ys, output)
		_, err := gorgonia.Grad(lossValue, params...)
		runtime.Assert(err)
	}
	return gorgonia.NewTapeMachine(m.g,
			gorgonia.BindDualValues(params...)),
		output, lossValue, params
}

func (m *model) Train(epoch int, loss loss.Loss, optimizer optimizer.Optimizer, x, y tensor.Tensor) {
	if epoch%clearSteps == 0 {
		m.hidden = nil
		m.cell = nil
		m.g = gorgonia.NewGraph()
	}
	vm, _, _, params := m.Forward(loss, x, y)
	defer vm.Close()
	runtime.Assert(vm.RunAll())
	runtime.Assert(optimizer.Step(params))
}

func (m *model) Predict(x tensor.Tensor) gorgonia.Value {
	vm, pred, _, _ := m.Forward(nil, x, nil)
	defer vm.Close()
	runtime.Assert(vm.RunAll())
	return pred.Value()
}

func (m *model) Loss(loss loss.Loss, x, y tensor.Tensor) float32 {
	vm, _, lossValue, _ := m.Forward(loss, x, y)
	defer vm.Close()
	runtime.Assert(vm.RunAll())
	return lossValue.Value().Data().(float32)
}

func (m *model) Params(g *gorgonia.ExprGraph) gorgonia.Nodes {
	var ret gorgonia.Nodes
	if m.rnn != nil {
		for name, p := range m.rnn.Params() {
			ret = append(ret, gorgonia.NodeFromAny(g, p, gorgonia.WithName(name)))
		}
	}
	if m.lstm != nil {
		for name, p := range m.lstm.Params() {
			ret = append(ret, gorgonia.NodeFromAny(g, p, gorgonia.WithName(name)))
		}
	}
	for name, p := range m.outputLayer.Params() {
		ret = append(ret, gorgonia.NodeFromAny(g, p, gorgonia.WithName(name)))
	}
	sort.Slice(ret, func(i, j int) bool {
		return ret[i].Hashcode() < ret[j].Hashcode()
	})
	return ret
}
