package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"gorgonia.org/gorgonia"
)

type Rnn struct {
	*base
	featureSize, steps int
	hidden             int
	// params
	Wih *gorgonia.Node
	Whh *gorgonia.Node
	Bih *gorgonia.Node
	Bhh *gorgonia.Node
}

func NewRnn(g *gorgonia.ExprGraph, featureSize, steps, hidden int) Layer {
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
	layer.Wih = loadParam(g, params["Wih"], "Wih")
	layer.Whh = loadParam(g, params["Whh"], "Whh")
	layer.Bih = loadParam(g, params["Bih"], "Bih")
	layer.Bhh = loadParam(g, params["Bhh"], "Bhh")
	return &layer
}

func buildRnnBlock(x *gorgonia.Node, nodes []*gorgonia.Node, names []string, featureSize, steps, hidden int) []*gorgonia.Node {
	if nodes[0] == nil {
		nodes[0] = gorgonia.NewTensor(x.Graph(), gorgonia.Float32, 3,
			gorgonia.WithShape(x.Shape()[0], featureSize, hidden),
			gorgonia.WithName(names[0]),
			gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	}
	if nodes[1] == nil {
		nodes[1] = gorgonia.NewTensor(x.Graph(), gorgonia.Float32, 3,
			gorgonia.WithShape(x.Shape()[0], hidden, hidden),
			gorgonia.WithName(names[1]),
			gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	}
	if nodes[2] == nil {
		nodes[2] = gorgonia.NewTensor(x.Graph(), gorgonia.Float32, 3,
			gorgonia.WithShape(x.Shape()[0], steps, hidden),
			gorgonia.WithName(names[2]),
			gorgonia.WithInit(gorgonia.Zeroes()))
	}
	if nodes[3] == nil {
		nodes[3] = gorgonia.NewTensor(x.Graph(), gorgonia.Float32, 3,
			gorgonia.WithShape(x.Shape()[0], steps, hidden),
			gorgonia.WithName(names[3]),
			gorgonia.WithInit(gorgonia.Zeroes()))
	}
	return nodes
}

func (layer *Rnn) Forward(x *gorgonia.Node, _ bool) *gorgonia.Node {
	block := buildRnnBlock(x, []*gorgonia.Node{layer.Wih, layer.Whh, layer.Bih, layer.Bhh},
		[]string{"Wih", "Whh", "Bih", "Bhh"}, layer.featureSize, layer.steps, layer.hidden)
	layer.Wih = block[0]
	layer.Whh = block[1]
	layer.Bih = block[2]
	layer.Bhh = block[3]
	h := gorgonia.NewTensor(x.Graph(), gorgonia.Float32, 3,
		gorgonia.WithShape(x.Shape()[0], layer.steps, layer.hidden), gorgonia.WithName("h"),
		gorgonia.WithInit(gorgonia.Zeroes()))
	for i := 0; i < layer.steps; i++ {
		a := gorgonia.Must(gorgonia.BatchedMatMul(x, layer.Wih))
		a = gorgonia.Must(gorgonia.Add(a, layer.Bih))
		b := gorgonia.Must(gorgonia.BatchedMatMul(h, layer.Whh))
		b = gorgonia.Must(gorgonia.Add(b, layer.Bhh))
		h = gorgonia.Must(gorgonia.Tanh(gorgonia.Must(gorgonia.Add(a, b))))
	}
	return h
}

func (layer *Rnn) Params() gorgonia.Nodes {
	return gorgonia.Nodes{layer.Wih, layer.Whh, layer.Bih, layer.Bhh}
}

func (layer *Rnn) Args() map[string]float32 {
	return map[string]float32{
		"feature_size": float32(layer.featureSize),
		"steps":        float32(layer.steps),
		"hidden":       float32(layer.hidden),
	}
}
