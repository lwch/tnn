package net

import (
	"fmt"

	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/params"
	"gonum.org/v1/gonum/mat"
)

type loadFunc func(name string, params map[string]*pb.Dense, args map[string]*pb.Dense) layer.Layer

var loadFuncs = map[string]loadFunc{
	"dense":   layer.LoadDense,
	"dropout": layer.LoadDropout,
	"conv2d":  layer.LoadConv2D,
	"maxpool": layer.LoadMaxPool,
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

func (n *Net) Add(layer layer.Layer) {
	n.layers = append(n.layers, layer)
}

func (n *Net) Forward(input mat.Matrix, isTraining bool) (ret mat.Matrix, context []mat.Matrix) {
	var ctx mat.Matrix
	list := make([]mat.Matrix, len(n.layers))
	for i := 0; i < len(n.layers); i++ {
		ctx, input = n.layers[i].Forward(input, isTraining)
		list[i] = ctx
		// fmt.Println(n.layers[i].Name())
		// fmt.Println(input.Dims())
	}
	return input, list
}

func (n *Net) Backward(grad mat.Matrix, ctx []mat.Matrix) []*params.Params {
	ret := make([]*params.Params, len(n.layers))
	for i := len(n.layers) - 1; i >= 0; i-- {
		var paramsGrad *params.Params
		grad, paramsGrad = n.layers[i].Backward(ctx[i], grad)
		// fmt.Println(n.layers[i].Name())
		// fmt.Println(grad.Dims())
		ret[i] = paramsGrad
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
		ps.Range(func(_ string, val mat.Matrix) {
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
		ret[i].Class = n.layers[i].Class()
		ret[i].Name = n.layers[i].Name()
		ps := n.layers[i].Params()
		if ps != nil && ps.Size() > 0 {
			ret[i].Params = make(map[string]*pb.Dense)
			ps.Range(func(key string, val mat.Matrix) {
				var dense pb.Dense
				rows, cols := val.Dims()
				dense.Rows, dense.Cols = int32(rows), int32(cols)
				dense.Data = mat.DenseCopyOf(val).RawMatrix().Data
				ret[i].Params[key] = &dense
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

func (n *Net) Print() {
	fmt.Println("Layers:")
	for i := 0; i < len(n.layers); i++ {
		n.layers[i].Print()
	}
}

func (n *Net) Layers() []layer.Layer {
	return n.layers
}
