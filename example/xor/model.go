package main

import (
	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"github.com/sugarme/gotch/nn"
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

func (m *model) Forward(vs *nn.Path, x *ts.Tensor) *ts.Tensor {
	output := x
	for _, l := range m.net.Layers() {
		switch ln := l.(type) {
		case *layer.Dense:
			output = ln.Forward(vs, output)
		case *activation.ReLU:
			output = ln.Forward(output)
		}
	}
	return output
}

func (m *model) Train(vs *nn.VarStore, x, y *ts.Tensor) float32 {
	pred := m.Forward(vs.Root(), x)
	loss := m.loss.Loss(y, pred)
	runtime.Assert(m.optimizer.Step(vs, loss))
	return loss.Vals().([]float32)[0]
}

func (m *model) Predict(vs *nn.VarStore, x *ts.Tensor) []float32 {
	return m.Forward(vs.Root(), x).Vals().([]float32)
}

func (m *model) Loss(vs *nn.VarStore, x, y *ts.Tensor) float32 {
	pred := m.Forward(vs.Root(), x)
	loss := m.loss.Loss(y, pred)
	return loss.Vals().([]float32)[0]
}
