package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Rnn struct {
	*base
	featureSize, steps int
	hidden             int
	// params
	w tensor.Tensor
	b tensor.Tensor
}

func NewRnn(featureSize, steps, hidden int) *Rnn {
	var layer Rnn
	layer.base = new("rnn")
	layer.featureSize = featureSize
	layer.steps = steps
	layer.hidden = hidden
	return &layer
}

func LoadRnn(g *gorgonia.ExprGraph, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Rnn
	layer.base = new("rnn")
	layer.name = name
	layer.featureSize = int(args["feature_size"])
	layer.steps = int(args["steps"])
	layer.hidden = int(args["hidden"])
	layer.w = loadParam(g, params["w"])
	layer.b = loadParam(g, params["b"])
	return &layer
}

func (layer *Rnn) Forward(x, h *gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, error) {
	inputShape := x.Shape()
	if layer.w == nil {
		layer.w = initW(layer.featureSize+layer.hidden, layer.hidden)
	}
	if layer.b == nil {
		layer.b = initB(inputShape[0], layer.hidden)
	}
	w := gorgonia.NodeFromAny(x.Graph(), layer.w, gorgonia.WithName("w"))
	b := gorgonia.NodeFromAny(x.Graph(), layer.b, gorgonia.WithName("b"))
	if h == nil {
		h = gorgonia.NewMatrix(x.Graph(), gorgonia.Float32,
			gorgonia.WithShape(inputShape[0], layer.hidden), gorgonia.WithName("h"),
			gorgonia.WithInit(gorgonia.Zeroes()))
	}
	x, err := gorgonia.Transpose(x, 1, 0, 2) // (steps, batch, feature)
	if err != nil {
		return nil, nil, err
	}
	var result *gorgonia.Node
	for i := 0; i < layer.steps; i++ {
		t, err := gorgonia.Slice(x, gorgonia.S(i)) // (batch, feature)
		if err != nil {
			return nil, nil, err
		}
		z := gorgonia.Must(gorgonia.Concat(1, h, t)) // (batch, feature+hidden)
		z = gorgonia.Must(gorgonia.Mul(z, w))        // (batch, hidden)
		z = gorgonia.Must(gorgonia.Add(z, b))        // (batch, hidden)
		h = gorgonia.Must(gorgonia.Tanh(z))          // (batch, hidden)
		if result == nil {
			result = h
		} else {
			result = gorgonia.Must(gorgonia.Concat(0, result, h))
		}
	}
	return gorgonia.Must(gorgonia.Reshape(result,
		tensor.Shape{inputShape[0], inputShape[1], layer.hidden})), h, nil
}

func (layer *Rnn) Params() map[string]tensor.Tensor {
	return map[string]tensor.Tensor{
		"w": layer.w,
		"b": layer.b,
	}
}

func (layer *Rnn) Args() map[string]float32 {
	return map[string]float32{
		"feature_size": float32(layer.featureSize),
		"steps":        float32(layer.steps),
		"hidden":       float32(layer.hidden),
	}
}
