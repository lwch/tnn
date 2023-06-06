package net

import (
	"bytes"
	"io"
	"os"

	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"google.golang.org/protobuf/proto"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type loadFunc func(g *gorgonia.ExprGraph, name string, params map[string]*pb.Dense, args map[string]float32) layer.Layer

var loadFuncs = map[string]loadFunc{
	"dense":   layer.LoadDense,
	"dropout": layer.LoadDropout,
	// "conv2d":  layer.LoadConv2D,
	// "maxpool": layer.LoadMaxPool,
	"rnn":  layer.LoadRnn,
	"lstm": layer.LoadLstm,
	// "self_attention": layer.LoadSelfAttention,
	"nor":     layer.LoadNor,
	"flatten": layer.LoadFlatten,
	// activation
	"sigmoid": activation.LoadSigmoid,
	// "softplus": activation.Load("softplus"),
	// "tanh":     activation.Load("tanh"),
	"relu": activation.LoadRelu,
	// "gelu": activation.Load("gelu"),
}

type Net struct {
	g      *gorgonia.ExprGraph
	layers []layer.Layer
}

func New(g *gorgonia.ExprGraph) *Net {
	return &Net{g: g}
}

func (n *Net) Add(layers ...layer.Layer) {
	n.layers = append(n.layers, layers...)
}

func (n *Net) Params() []map[string]tensor.Tensor {
	var ret []map[string]tensor.Tensor
	for _, l := range n.layers {
		params := make(map[string]tensor.Tensor)
		for name, p := range l.Params() {
			params[name] = p
		}
		ret = append(ret, params)
	}
	return ret
}

func (n *Net) ParamCount() uint64 {
	var ret uint64
	for _, l := range n.layers {
		for _, p := range l.Params() {
			ret += uint64(p.Shape().TotalSize())
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
			shape := p.Shape()
			dense.Shape = make([]int32, len(shape))
			for j := 0; j < len(shape); j++ {
				dense.Shape[j] = int32(shape[j])
			}
			dense.Data = p.Data().([]float32)
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
		n.layers[i] = fn(n.g, name, layers[i].GetParams(), layers[i].GetArgs())
	}
	return int64(len(data)), nil
}

func (n *Net) Layers() []layer.Layer {
	return n.layers
}
