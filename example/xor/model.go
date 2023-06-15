package main

import (
	"github.com/lwch/gotorch/optimizer"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/net"
)

type model struct {
	net       *net.Net
	optimizer optimizer.Optimizer
}

func newModel(net *net.Net, optimizer optimizer.Optimizer) *model {
	return &model{
		net:       net,
		optimizer: optimizer,
	}
}

func (m *model) Forward(x *tensor.Tensor) *tensor.Tensor {
	output := x
	for _, l := range m.net.Layers() {
		switch ln := l.(type) {
		case *layer.Dense:
			output = ln.Forward(output)
		case *activation.ReLU:
			output = ln.Forward(output)
		}
	}
	return output
}

func (m *model) Train(x, y *tensor.Tensor) float32 {
	pred := m.Forward(x)
	l := lossFunc(pred, y)
	l.Backward()
	value := l.Value()
	m.optimizer.Step(m.net.Params())
	storage.GC()
	return float32(value)
}

func (m *model) Predict(x *tensor.Tensor) []float32 {
	return m.Forward(x).Float32Value()
}

func (m *model) Loss(x, y *tensor.Tensor) float32 {
	pred := m.Forward(x)
	loss := lossFunc(y, pred)
	return float32(loss.Value())
}
