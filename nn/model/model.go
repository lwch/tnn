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
	return m.net.Forward(input, false)
}

func filterEmptyParams(arr []*params.Params) []*params.Params {
	ret := make([]*params.Params, 0, len(arr))
	for i := 0; i < len(arr); i++ {
		if arr[i] == nil {
			continue
		}
		ret = append(ret, arr[i])
	}
	return ret
}

func (m *Model) Train(input, targets *tensor.Tensor) {
	pred := m.net.Forward(input, true)
	loss := m.loss.Loss(pred, targets)
	loss.ZeroGrad()
	loss.Backward(loss)
	m.optimizer.Update(filterEmptyParams(m.net.Params()))
	m.trainCount++
}

func (m *Model) Loss(input, targets *tensor.Tensor) float64 {
	pred := m.Predict(input)
	return m.loss.Loss(pred, targets).Value().At(0, 0)
}

func (m *Model) ParamCount() uint64 {
	return m.net.ParamCount()
}
