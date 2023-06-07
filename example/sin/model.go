package main

import (
	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/optimizer"
	"github.com/sugarme/gotch/ts"
)

type model struct {
	rnn         *layer.Rnn
	lstm        *layer.Lstm
	flatten     *layer.Flatten
	outputLayer *layer.Dense
	hidden      *ts.Tensor
	cell        *ts.Tensor
	loss        loss.Loss
	optimizer   optimizer.Optimizer
}

func newModel(loss loss.Loss, optimizer optimizer.Optimizer) *model {
	return &model{
		// rnn:         layer.NewRnn(featureSize, steps, hiddenSize),
		lstm:        layer.NewLstm(featureSize, steps, hiddenSize),
		flatten:     layer.NewFlatten(),
		outputLayer: layer.NewDense(1),
		loss:        loss,
		optimizer:   optimizer,
	}
}

func (m *model) Forward(x *ts.Tensor) *ts.Tensor {
	var output *ts.Tensor
	if m.rnn != nil {
		output, m.hidden = m.rnn.Forward(vs.Root(), x, m.hidden)
	} else {
		output, m.hidden, m.cell = m.lstm.Forward(vs.Root(), x, m.hidden, m.cell)
	}
	output = m.flatten.Forward(output)
	return m.outputLayer.Forward(vs.Root(), output)
}

func (m *model) Train(epoch int, x, y *ts.Tensor) float32 {
	m.hidden = nil // TODO: fix backward
	m.cell = nil   // TODO: fix backward
	pred := m.Forward(x)
	l := m.loss.Loss(y, pred)
	runtime.Assert(m.optimizer.Step(vs, l))
	return l.Vals().([]float32)[0]
}

func (m *model) Predict(x *ts.Tensor) []float32 {
	return m.Forward(x).Vals().([]float32)
}

func (m *model) Loss(x, y *ts.Tensor) float32 {
	pred := m.Forward(x)
	return m.loss.Loss(y, pred).Vals().([]float32)[0]
}
