package model

import (
	"tnn/internal/nn/loss"
	"tnn/internal/nn/net"
	"tnn/internal/nn/optimizer"

	"gonum.org/v1/gonum/mat"
)

type Model struct {
	net       *net.Net
	loss      loss.Loss
	optimizer optimizer.Optimizer
}

func New(net *net.Net, loss loss.Loss, optimizer optimizer.Optimizer) *Model {
	return &Model{
		net:       net,
		loss:      loss,
		optimizer: optimizer,
	}
}

func (m *Model) Predict(input *mat.Dense) *mat.Dense {
	return m.net.Forward(input)
}

func (m *Model) Train(input, targets *mat.Dense) float64 {
	t := m.Predict(input)
	loss := m.loss.Loss(t, targets)
	grad := m.loss.Grad(t, targets)
	m.net.Backward(grad)
	m.net.Update(m.optimizer)
	return loss
}
