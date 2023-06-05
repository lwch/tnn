package net

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"gorgonia.org/gorgonia"
)

type loadFunc func(g *gorgonia.ExprGraph, name string, params map[string]*pb.Dense, args map[string]float32) layer.Layer

var loadFuncs = map[string]loadFunc{
	"dense": layer.LoadDense,
	// "dropout": layer.LoadDropout,
	// "conv2d":  layer.LoadConv2D,
	// "maxpool": layer.LoadMaxPool,
	// "rnn":            layer.LoadRnn,
	// "lstm":           layer.LoadLstm,
	// "self_attention": layer.LoadSelfAttention,
	// "nor":            layer.LoadNor,
	"flatten": layer.LoadFlatten,
	// activation
	// "sigmoid":  activation.Load("sigmoid"),
	// "softplus": activation.Load("softplus"),
	// "tanh":     activation.Load("tanh"),
	"relu": activation.Load("relu"),
	// "gelu": activation.Load("gelu"),
}

type Net struct {
	layers []layer.Layer
}

func New(layers ...layer.Layer) *Net {
	return &Net{
		layers: layers,
	}
}

func (n *Net) Add(layers ...layer.Layer) {
	n.layers = append(n.layers, layers...)
}

func (n *Net) Forward(x *gorgonia.Node) *gorgonia.Node {
	for _, l := range n.layers {
		x = l.Forward(x)
	}
	return x
}

func (n *Net) Params() gorgonia.Nodes {
	var ret gorgonia.Nodes
	for _, l := range n.layers {
		ret = append(ret, l.Params()...)
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

func (n *Net) Save() []*pb.Layer {
	ret := make([]*pb.Layer, len(n.layers))
	for i := 0; i < len(n.layers); i++ {
		ret[i] = new(pb.Layer)
		ret[i].Class = n.layers[i].Class()
		ret[i].Name = n.layers[i].Name()
		ret[i].Params = make(map[string]*pb.Dense)
		for _, p := range n.layers[i].Params() {
			var dense pb.Dense
			shape := p.Shape()
			dense.Shape = make([]int32, len(shape))
			for j := 0; j < len(shape); j++ {
				dense.Shape[j] = int32(shape[j])
			}
			dense.Data = p.Value().Data().([]float32)
			ret[i].Params[p.Name()] = &dense
		}
		ret[i].Args = n.layers[i].Args()
	}
	return ret
}

func (n *Net) Load(g *gorgonia.ExprGraph, layers []*pb.Layer) {
	n.layers = make([]layer.Layer, len(layers))
	for i := 0; i < len(layers); i++ {
		class := layers[i].GetClass()
		fn := loadFuncs[class]
		if fn == nil {
			panic("unsupported " + class + " layer")
		}
		name := layers[i].GetName()
		n.layers[i] = fn(g, name, layers[i].GetParams(), layers[i].GetArgs())
	}
}
