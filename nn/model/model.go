package model

import (
	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"gorgonia.org/gorgonia"
)

type Model struct {
	vm        gorgonia.VM
	net       *net.Net
	loss      loss.Loss
	lossValue gorgonia.Value
	optimizer optimizer.Optimizer
}

func New(net *net.Net, loss loss.Loss, optimizer optimizer.Optimizer) *Model {
	return &Model{
		net:       net,
		loss:      loss,
		optimizer: optimizer,
	}
}

func (m *Model) Close() {
	if m.vm != nil {
		m.vm.Close()
	}
}

func (m *Model) Compile(g *gorgonia.ExprGraph, x, y *gorgonia.Node) *gorgonia.Node {
	pred := m.net.Forward(x)
	loss := m.loss.Loss(y, pred)
	gorgonia.Read(loss, &m.lossValue)
	_, err := gorgonia.Grad(loss, m.net.Params()...)
	runtime.Assert(err)
	if m.vm != nil {
		m.vm.Close()
	}
	m.vm = gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.net.Params()...))
	return pred
}

func (m *Model) Train() error {
	m.vm.Reset()
	err := m.vm.RunAll()
	if err != nil {
		return err
	}
	return m.optimizer.Step(m.net.Params())
}

func (m *Model) Loss() float32 {
	return m.lossValue.Data().(float32)
}

func (m *Model) ParamCount() int {
	return m.net.ParamCount()
}

func (m *Model) RunAll() error {
	return m.vm.RunAll()
}
