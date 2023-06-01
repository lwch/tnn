package model

import (
	"bytes"
	"io"
	"os"

	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
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
		name:      "<unset>",
		net:       net,
		loss:      loss,
		optimizer: optimizer,
	}
}

func (m *Model) Forward(input *tensor.Tensor, train bool) *tensor.Tensor {
	return m.net.Forward(input, train)
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

func (m *Model) Backward(pred, targets *tensor.Tensor) {
	loss := m.loss.Loss(pred, targets)
	loss.Backward(loss)
}

func (m *Model) Loss(input, targets *tensor.Tensor) float32 {
	pred := m.Forward(input, false)
	return m.loss.Loss(pred, targets).Value().At(0, 0)
}

func (m *Model) Apply() {
	m.optimizer.Update(filterEmptyParams(m.net.Params()))
	m.trainCount++
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
	// if m.lr != nil {
	// 	model.Scheduler = m.lr.Save()
	// }
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
	// if model.GetScheduler() != nil {
	// 	m.lr = lr.Load(model.GetScheduler(), m.optimizer)
	// }
	return int64(len(data)), nil
}

func (m *Model) Layers() []layer.Layer {
	return m.net.Layers()
}
