package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Dense struct {
	*base
	output int
	// params
	w tensor.Tensor
	b tensor.Tensor
}

func NewDense(output int) *Dense {
	var layer Dense
	layer.base = new("dense")
	layer.output = output
	return &layer
}

func LoadDense(g *gorgonia.ExprGraph, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Dense
	layer.base = new("dense")
	layer.name = name
	layer.output = int(args["output"])
	layer.w = loadParam(g, params["w"])
	layer.b = loadParam(g, params["b"])
	return &layer
}

func (layer *Dense) Forward(x *gorgonia.Node) (*gorgonia.Node, gorgonia.Nodes) {
	if layer.w == nil {
		layer.w = initW(x.Shape()[1], layer.output)
	}
	if layer.b == nil {
		layer.b = initB(x.Shape()[0], layer.output)
	}
	w := gorgonia.NodeFromAny(x.Graph(), layer.w, gorgonia.WithName("w"))
	b := gorgonia.NodeFromAny(x.Graph(), layer.b, gorgonia.WithName("b"))
	wx := gorgonia.Must(gorgonia.Mul(x, w))
	return gorgonia.Must(gorgonia.Add(wx, b)),
		gorgonia.Nodes{w, b}
}

func (layer *Dense) Params() map[string]tensor.Tensor {
	return map[string]tensor.Tensor{
		"w": layer.w,
		"b": layer.b,
	}
}

func (layer *Dense) Args() map[string]float32 {
	return map[string]float32{
		"output": float32(layer.output),
	}
}
