package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"gorgonia.org/gorgonia"
)

type Dense struct {
	*base
	input, output int
	// params
	w *gorgonia.Node
	b *gorgonia.Node
}

func NewDense(g *gorgonia.ExprGraph, input, output int) Layer {
	var layer Dense
	layer.base = new("dense")
	layer.input = input
	layer.output = output
	layer.w = gorgonia.NewMatrix(g, gorgonia.Float32,
		gorgonia.WithShape(input, output),
		gorgonia.WithName("w"),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	return &layer
}

func LoadDense(g *gorgonia.ExprGraph, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Dense
	layer.base = new("dense")
	layer.name = name
	layer.input = int(args["input"])
	layer.output = int(args["output"])
	layer.w = loadParam(g, params["w"], "w")
	layer.b = loadParam(g, params["b"], "b")
	return &layer
}

func (layer *Dense) Forward(x *gorgonia.Node) *gorgonia.Node {
	if layer.b == nil {
		layer.b = gorgonia.NewMatrix(x.Graph(), gorgonia.Float32,
			gorgonia.WithShape(x.Shape()[0], layer.output),
			gorgonia.WithName("b"),
			gorgonia.WithInit(gorgonia.Zeroes()))
	}
	wx := gorgonia.Must(gorgonia.Mul(x, layer.w))
	return gorgonia.Must(gorgonia.Add(wx, layer.b))
}

func (layer *Dense) Params() gorgonia.Nodes {
	return gorgonia.Nodes{layer.w, layer.b}
}

func (layer *Dense) Args() map[string]float32 {
	return map[string]float32{
		"input":  float32(layer.input),
		"output": float32(layer.output),
	}
}
