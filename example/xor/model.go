package main

import (
	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"github.com/sugarme/gotch/ts"
)

type model struct {
	net       *net.Net
	loss      loss.Loss
	optimizer optimizer.Optimizer
}

func newModel(net *net.Net, loss loss.Loss, optimizer optimizer.Optimizer) *model {
	return &model{
		net:       net,
		loss:      loss,
		optimizer: optimizer,
	}
}

func (m *model) Forward(x *ts.Tensor) *ts.Tensor {
	output := x
	for _, l := range m.net.Layers() {
		switch ln := l.(type) {
		case *layer.Dense:
			output = ln.Forward(vs.Root(), output)
		case *activation.ReLU:
			output = ln.Forward(output)
		}
	}
	return output
}

func (m *model) Train(x, y *ts.Tensor) float32 {
	pred := m.Forward(x)
	loss := m.loss.Loss(y, pred)
	runtime.Assert(m.optimizer.Step(vs, loss))
	return loss.Vals().([]float32)[0]
}

func (m *model) Predict(x *ts.Tensor) []float32 {
	return m.Forward(x).Vals().([]float32)
}

func (m *model) Loss(x, y *ts.Tensor) float32 {
	pred := m.Forward(x)
	loss := m.loss.Loss(y, pred)
	return loss.Vals().([]float32)[0]
}
