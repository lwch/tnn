package main

import (
	"github.com/lwch/gotorch/optimizer"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
)

type model struct {
	rnn         *layer.Rnn
	lstm        *layer.Lstm
	flatten     *layer.Flatten
	outputLayer *layer.Linear
	hidden      *tensor.Tensor
	cell        *tensor.Tensor
	optimizer   optimizer.Optimizer
}

func newModel(optimizer optimizer.Optimizer) *model {
	return &model{
		rnn: layer.NewRnn("rnn", featureSize, steps, hiddenSize, layer.WithDevice(device)),
		// lstm:        layer.NewLstm("lstm", featureSize, steps, hiddenSize, layer.WithDevice(device)),
		flatten:     layer.NewFlatten("flatten"),
		outputLayer: layer.NewLinear("output", steps*hiddenSize, 1, layer.WithDevice(device)),
		optimizer:   optimizer,
	}
}

func (m *model) Forward(x *tensor.Tensor, train bool) *tensor.Tensor {
	var output *tensor.Tensor
	if m.rnn != nil {
		var hidden *tensor.Tensor
		output, hidden = m.rnn.Forward(x, m.hidden)
		if train {
			m.hidden = hidden
		}
	} else {
		var hidden, cell *tensor.Tensor
		output, hidden, cell = m.lstm.Forward(x, m.hidden, m.cell)
		if train {
			m.hidden = hidden
			m.cell = cell
		}
	}
	output = m.flatten.Forward(output)
	return m.outputLayer.Forward(output)
}

func (m *model) Train(epoch int, x, y *tensor.Tensor) float32 {
	pred := m.Forward(x, true)
	l := lossFunc(pred, y)
	l.Backward()
	value := l.Value()
	m.optimizer.Step(m.params())
	return float32(value)
}

func (m *model) Predict(x *tensor.Tensor) []float32 {
	return m.Forward(x, false).Float32Value()
}

func (m *model) Loss(x, y *tensor.Tensor) float32 {
	pred := m.Forward(x, false)
	return float32(lossFunc(pred, y).Value())
}

func (m *model) params() []*tensor.Tensor {
	var ret []*tensor.Tensor
	if m.rnn != nil {
		for _, p := range m.rnn.Params() {
			ret = append(ret, p)
		}
	}
	if m.lstm != nil {
		for _, p := range m.lstm.Params() {
			ret = append(ret, p)
		}
	}
	for _, p := range m.outputLayer.Params() {
		ret = append(ret, p)
	}
	return ret
}
