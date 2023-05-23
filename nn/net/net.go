package net

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
)

type loadFunc func(name string, params map[string]*pb.Dense, args map[string]*pb.Dense) layer.Layer

var loadFuncs = map[string]loadFunc{
	"dense": layer.LoadDense,
	// "dropout": layer.LoadDropout,
	// "conv2d":  layer.LoadConv2D,
	// "maxpool": layer.LoadMaxPool,
	"rnn": layer.LoadRnn,
	// activation
	"sigmoid":  activation.Load("sigmoid"),
	"softplus": activation.Load("softplus"),
	"tanh":     activation.Load("tanh"),
	"relu":     activation.Load("relu"),
}

type Net struct {
	layers []layer.Layer
}

func New() *Net {
	return &Net{}
}

func (n *Net) Set(layer ...layer.Layer) {
	n.layers = layer
}

func (n *Net) Forward(input *tensor.Tensor, isTraining bool) *tensor.Tensor {
	for i := 0; i < len(n.layers); i++ {
		input = n.layers[i].Forward(input, isTraining)
	}
	return input
}

func (n *Net) Params() []*params.Params {
	ret := make([]*params.Params, len(n.layers))
	for i := 0; i < len(n.layers); i++ {
		ret[i] = n.layers[i].Params()
	}
	return ret
}

func (n *Net) ParamCount() uint64 {
	var count uint64
	for i := 0; i < len(n.layers); i++ {
		params := n.layers[i].Params()
		if params == nil {
			continue
		}
		params.Range(func(_ string, dense *tensor.Tensor) {
			rows, cols := dense.Dims()
			count += uint64(rows * cols)
		})
	}
	return count
}

func (n *Net) SaveLayers() []*pb.Layer {
	ret := make([]*pb.Layer, len(n.layers))
	for i := 0; i < len(n.layers); i++ {
		ret[i] = new(pb.Layer)
		ret[i].Class = n.layers[i].Class()
		ret[i].Name = n.layers[i].Name()
		ps := n.layers[i].Params()
		if ps != nil && ps.Size() > 0 {
			ret[i].Params = make(map[string]*pb.Dense)
			ps.Range(func(name string, t *tensor.Tensor) {
				var dense pb.Dense
				rows, cols := t.Dims()
				dense.Rows, dense.Cols = int32(rows), int32(cols)
				dense.Data = t.Clone().Value().RawMatrix().Data
				ret[i].Params[name] = &dense
			})
		}
		args := n.layers[i].Args()
		if len(args) > 0 {
			ret[i].Args = make(map[string]*pb.Dense)
			for k, v := range args {
				var dense pb.Dense
				rows, cols := v.Dims()
				dense.Rows, dense.Cols = int32(rows), int32(cols)
				dense.Data = mat.DenseCopyOf(v).RawMatrix().Data
				ret[i].Args[k] = &dense
			}
		}
	}
	return ret
}

func (n *Net) LoadLayers(layers []*pb.Layer) {
	n.layers = make([]layer.Layer, len(layers))
	for i := 0; i < len(layers); i++ {
		class := layers[i].GetClass()
		fn := loadFuncs[class]
		if fn == nil {
			panic("unsupported " + class + " layer")
		}
		name := layers[i].GetName()
		n.layers[i] = fn(name, layers[i].GetParams(), layers[i].GetArgs())
	}
}
