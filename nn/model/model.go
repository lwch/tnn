package model

import (
	"bytes"
	"io"
	"os"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"google.golang.org/protobuf/proto"
	"gorgonia.org/gorgonia"
)

type Model struct {
	name       string
	trainCount uint64
	vm         gorgonia.VM
	net        *net.Net
	loss       loss.Loss
	lossValue  gorgonia.Value
	optimizer  optimizer.Optimizer
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
	err = m.optimizer.Step(m.net.Params())
	if err != nil {
		return err
	}
	m.trainCount++
	return nil
}

func (m *Model) Loss() float32 {
	return m.lossValue.Data().(float32)
}

func (m *Model) ParamCount() uint64 {
	return m.net.ParamCount()
}

func (m *Model) RunAll() error {
	return m.vm.RunAll()
}

func (m *Model) Save(dir string) error {
	f, err := os.Create(dir)
	if err != nil {
		return err
	}
	_, err = m.WriteTo(f)
	return err
}

func (m *Model) WriteTo(w io.Writer) (int64, error) {
	var model pb.Model
	model.Name = m.name
	model.TrainCount = m.trainCount
	model.ParamCount = m.ParamCount()
	model.Layers = m.net.Save()
	model.Loss = m.loss.Save()
	model.Optimizer = m.optimizer.Save()
	// if m.lr != nil {
	// 	model.Scheduler = m.lr.Save()
	// }
	data, err := proto.Marshal(&model)
	if err != nil {
		return 0, err
	}
	return io.Copy(w, bytes.NewReader(data))
}

func (m *Model) Load(g *gorgonia.ExprGraph, dir string) error {
	f, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = m.ReadFrom(g, f)
	return err
}

func (m *Model) ReadFrom(g *gorgonia.ExprGraph, r io.Reader) (int64, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return 0, err
	}
	var model pb.Model
	err = proto.Unmarshal(data, &model)
	if err != nil {
		return 0, err
	}
	m.name = model.GetName()
	m.trainCount = model.GetTrainCount()
	m.net = net.New()
	m.net.Load(g, model.GetLayers())
	m.loss = loss.Load(model.GetLoss())
	m.optimizer = optimizer.Load(model.GetOptimizer())
	// if model.GetScheduler() != nil {
	// 	m.lr = lr.Load(model.GetScheduler(), m.optimizer)
	// }
	return int64(len(data)), nil
}
