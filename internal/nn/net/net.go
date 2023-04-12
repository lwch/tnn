package net

import (
	"tnn/internal/nn/layer"
	"tnn/internal/nn/layer/activation"
	"tnn/internal/nn/params"
	"tnn/internal/nn/pb"

	"gonum.org/v1/gonum/mat"
)

type loadFunc func(map[string]*pb.Dense) layer.Layer

var loadFuncs = map[string]loadFunc{
	"dense":   layer.LoadDense,
	"sigmoid": activation.Load("sigmoid"),
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

func (n *Net) Add(layer layer.Layer) {
	n.layers = append(n.layers, layer)
}

func (n *Net) Forward(input *mat.Dense) *mat.Dense {
	for i := 0; i < len(n.layers); i++ {
		input = n.layers[i].Forward(input)
	}
	return input
}

func (n *Net) Backward(grad *mat.Dense) []*params.Params {
	ret := make([]*params.Params, len(n.layers))
	for i := len(n.layers) - 1; i >= 0; i-- {
		grad = n.layers[i].Backward(grad)
		var p params.Params
		p.Copy(n.layers[i].Context())
		ret[i] = &p
	}
	return ret
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
		ps := n.layers[i].Params()
		if ps == nil {
			continue
		}
		ps.Range(func(_ string, val *mat.Dense) {
			rows, cols := val.Dims()
			count += uint64(rows * cols)
		})
	}
	return count
}

func (n *Net) SaveLayers() []*pb.Layer {
	ret := make([]*pb.Layer, len(n.layers))
	for i := 0; i < len(n.layers); i++ {
		ret[i] = new(pb.Layer)
		ret[i].Name = n.layers[i].Name()
		ps := n.layers[i].Params()
		if ps == nil {
			continue
		}
		ret[i].Params = make(map[string]*pb.Dense)
		ps.Range(func(key string, val *mat.Dense) {
			var dense pb.Dense
			rows, cols := val.Dims()
			dense.Rows, dense.Cols = int32(rows), int32(cols)
			dense.Data = mat.DenseCopyOf(val).RawMatrix().Data
			ret[i].Params[key] = &dense
		})
	}
	return ret
}

func (n *Net) LoadLayers(layers []*pb.Layer) {
	n.layers = make([]layer.Layer, len(layers))
	for i := 0; i < len(layers); i++ {
		name := layers[i].GetName()
		fn := loadFuncs[name]
		if fn == nil {
			panic("unsupported " + name + " layer")
		}
		n.layers[i] = fn(layers[i].GetParams())
	}
}
