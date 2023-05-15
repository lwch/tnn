package model

import (
	"bytes"
	"fmt"
	"io"
	"os"

	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/lr"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"github.com/lwch/tnn/nn/params"
	"gonum.org/v1/gonum/mat"
	"google.golang.org/protobuf/proto"
)

type Model struct {
	name       string
	trainCount uint64
	net        *net.Net
	loss       loss.Loss
	optimizer  optimizer.Optimizer
	lr         lr.Scheduler
}

func New(net *net.Net, loss loss.Loss, optimizer optimizer.Optimizer) *Model {
	return &Model{
		name:      "<unset>",
		net:       net,
		loss:      loss,
		optimizer: optimizer,
	}
}

func (m *Model) SetName(name string) {
	m.name = name
}

func (m *Model) SetLrScheduler(lr lr.Scheduler) {
	m.lr = lr
}

func (m *Model) Predict(input mat.Matrix) mat.Matrix {
	result, _ := m.net.Forward(input, false)
	return result
}

func (m *Model) Train(input mat.Matrix, targets mat.Matrix) {
	pred, ctx := m.net.Forward(input, true)
	grad := m.loss.Grad(pred, targets)
	grads := m.net.Backward(grad, ctx)
	m.apply(grads)
	m.trainCount++
}

func (m *Model) Loss(input, targets mat.Matrix) float64 {
	pred := m.Predict(input)
	return m.loss.Loss(pred, targets)
}

func (m *Model) apply(grads []*params.Params) {
	params := m.net.Params()
	if m.lr != nil {
		lr := m.lr.Step(m.optimizer.GetLr())
		m.optimizer.Update(lr, grads, params)
		return
	}
	m.optimizer.Update(0, grads, params)
}

func (m *Model) ParamCount() uint64 {
	return m.net.ParamCount()
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
	model.Layers = m.net.SaveLayers()
	model.Loss = m.loss.Save()
	model.Optimizer = m.optimizer.Save()
	if m.lr != nil {
		model.Scheduler = m.lr.Save()
	}
	data, err := proto.Marshal(&model)
	if err != nil {
		return 0, err
	}
	return io.Copy(w, bytes.NewReader(data))
}

func (m *Model) Load(dir string) error {
	f, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = m.ReadFrom(f)
	return err
}

func (m *Model) ReadFrom(r io.Reader) (int64, error) {
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
	m.net.LoadLayers(model.GetLayers())
	m.loss = loss.Load(model.GetLoss())
	m.optimizer = optimizer.Load(model.GetOptimizer())
	if model.GetScheduler() != nil {
		m.lr = lr.Load(model.GetScheduler(), m.optimizer)
	}
	return int64(len(data)), nil
}

func (m *Model) Print() {
	fmt.Println("Model:", m.name)
	fmt.Println("Train Count:", m.trainCount)
	fmt.Println("Param Count:", m.ParamCount())
	loss.Print(m.loss)
	m.optimizer.Print()
	m.net.Print()
}

func (m *Model) Layers() []layer.Layer {
	return m.net.Layers()
}

func (m *Model) SetLr(lr float64) {
	m.optimizer.SetLr(lr)
}
