package model

import (
	"bytes"
	"io"
	"os"
	"tnn/internal/nn/loss"
	"tnn/internal/nn/net"
	"tnn/internal/nn/optimizer"
	"tnn/internal/nn/params"
	"tnn/internal/nn/pb"

	"gonum.org/v1/gonum/mat"
	"google.golang.org/protobuf/proto"
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
		name:      "<unset model name>",
		net:       net,
		loss:      loss,
		optimizer: optimizer,
	}
}

func (m *Model) SetName(name string) {
	m.name = name
}

func (m *Model) Predict(input *mat.Dense) *mat.Dense {
	return m.net.Forward(input)
}

func (m *Model) Train(input, targets *mat.Dense) {
	pred := m.Predict(input)
	grad := m.loss.Grad(pred, targets)
	grads := m.net.Backward(grad)
	m.apply(grads)
	m.trainCount++
}

func (m *Model) Loss(input, targets *mat.Dense) float64 {
	pred := m.Predict(input)
	return m.loss.Loss(pred, targets)
}

func (m *Model) apply(grads []*params.Params) {
	params := m.net.Params()
	m.optimizer.Update(grads, params)
}

func (m *Model) Save(dir string) error {
	var model pb.Model
	model.Name = m.name
	model.TrainCount = m.trainCount
	model.Layers = m.net.SaveLayers()
	model.Loss = m.loss.Save()
	model.Optimizer = m.optimizer.Save()
	f, err := os.Create(dir)
	if err != nil {
		return err
	}
	data, err := proto.Marshal(&model)
	if err != nil {
		return err
	}
	_, err = io.Copy(f, bytes.NewReader(data))
	return err
}

func (m *Model) Load(dir string) error {
	return nil
}
