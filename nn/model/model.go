package model

import (
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
)

type Model struct {
	name       string
	trainCount uint64
	net        *net.Net
	loss       loss.Loss
	optimizer  optimizer.Optimizer
}

func New(net *net.Net, loss loss.Loss, optimizer optimizer.Optimizer) *Model {
	return &Model{
		name:      "<unset>",
		net:       net,
		loss:      loss,
		optimizer: optimizer,
	}
}

func (m *Model) Predict(input *tensor.Tensor) *tensor.Tensor {
	return m.net.Forward(input, nil, false)
}

func (m *Model) Train(input, targets *tensor.Tensor) {
	watchList := params.NewList()
	pred := m.net.Forward(input, watchList, true)
	loss := m.loss.Loss(pred, targets)
	loss.ZeroGrad()
	loss.Backward(loss)
	m.optimizer.Update(watchList)
	m.trainCount++
}

func (m *Model) Loss(input, targets *tensor.Tensor) float64 {
	pred := m.Predict(input)
	return m.loss.Loss(pred, targets).Value().At(0, 0)
}
