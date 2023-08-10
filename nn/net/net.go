package net

import (
	"bytes"
	"io"
	"os"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"google.golang.org/protobuf/proto"
)

type loadFunc func(device consts.DeviceType, name string, params map[string]*pb.Dense, args map[string]float32) layer.Layer

var loadFuncs = map[string]loadFunc{
	"dense":      layer.LoadDense,
	"dropout":    layer.LoadDropout,
	"conv1d":     layer.LoadConv1D,
	"maxpool1d":  layer.LoadMaxPool1D,
	"rnn":        layer.LoadRnn,
	"lstm":       layer.LoadLstm,
	"attention":  layer.LoadAttention,
	"layer_norm": layer.LoadLayerNorm,
	"flatten":    layer.LoadFlatten,
	"embedding":  layer.LoadEmbedding,
	"rezero":     layer.LoadReZero,
	// activation
	"sigmoid": activation.LoadSigmoid,
	"tanh":    activation.LoadTanh,
	"relu":    activation.LoadRelu,
	"gelu":    activation.LoadGelu,
}

type Net struct {
	layers []layer.Layer
	device consts.DeviceType
}

func New(device consts.DeviceType) *Net {
	return &Net{device: device}
}

func (n *Net) SetDevice(device consts.DeviceType) {
	n.device = device
}

func (n *Net) Add(layers ...layer.Layer) {
	n.layers = append(n.layers, layers...)
}

func (n *Net) Params() []*tensor.Tensor {
	var ret []*tensor.Tensor
	for _, l := range n.layers {
		for _, p := range l.Params() {
			ret = append(ret, p)
		}
	}
	return ret
}

func (n *Net) ParamCount() uint64 {
	var ret uint64
	for _, l := range n.layers {
		for _, p := range l.Params() {
			ret += uint64(p.ElemCount())
		}
	}
	return ret
}

func (n *Net) Save(dir string) error {
	f, err := os.Create(dir)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = n.WriteTo(f)
	return err
}

func (n *Net) WriteTo(w io.Writer) (int64, error) {
	var net pb.Net
	net.Layers = make([]*pb.Layer, len(n.layers))
	for i := 0; i < len(n.layers); i++ {
		net.Layers[i] = new(pb.Layer)
		net.Layers[i].Class = n.layers[i].Class()
		net.Layers[i].Name = n.layers[i].Name()
		net.Layers[i].Params = make(map[string]*pb.Dense)
		for name, p := range n.layers[i].Params() {
			var dense pb.Dense
			shape := p.Shapes()
			dense.Shape = make([]int32, len(shape))
			for j := 0; j < len(shape); j++ {
				dense.Shape[j] = int32(shape[j])
			}
			dense.Data = p.ToDevice(consts.KCPU).Float32Value()
			net.Layers[i].Params[name] = &dense
		}
		net.Layers[i].Args = n.layers[i].Args()
	}
	data, err := proto.Marshal(&net)
	if err != nil {
		return 0, err
	}
	return io.Copy(w, bytes.NewReader(data))
}

func (n *Net) Load(dir string) error {
	f, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = n.ReadFrom(f)
	return err
}

func (n *Net) ReadFrom(r io.Reader) (int64, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return 0, err
	}
	var net pb.Net
	if err = proto.Unmarshal(data, &net); err != nil {
		return 0, err
	}
	layers := net.GetLayers()
	n.layers = make([]layer.Layer, len(layers))
	for i := 0; i < len(layers); i++ {
		class := layers[i].GetClass()
		fn := loadFuncs[class]
		if fn == nil {
			panic("unsupported " + class + " layer")
		}
		name := layers[i].GetName()
		n.layers[i] = fn(n.device, name, layers[i].GetParams(), layers[i].GetArgs())
	}
	return int64(len(data)), nil
}

func (n *Net) Layers() []layer.Layer {
	return n.layers
}
