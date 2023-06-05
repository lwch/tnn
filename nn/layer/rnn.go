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
	Wih := loadParam(params["Wih"])
	Whh := loadParam(params["Whh"])
	Bih := loadParam(params["Bih"])
	Bhh := loadParam(params["Bhh"])
	layer.Wih = gorgonia.NewMatrix(g, gorgonia.Float32,
		gorgonia.WithShape(Wih.Shape()...),
		gorgonia.WithName("Wih"))
	layer.Whh = gorgonia.NewMatrix(g, gorgonia.Float32,
		gorgonia.WithShape(Whh.Shape()...),
		gorgonia.WithName("Whh"))
	layer.Bih = gorgonia.NewMatrix(g, gorgonia.Float32,
		gorgonia.WithShape(Bih.Shape()...),
		gorgonia.WithName("Bih"))
	layer.Bhh = gorgonia.NewMatrix(g, gorgonia.Float32,
		gorgonia.WithShape(Bhh.Shape()...),
		gorgonia.WithName("Bhh"))
	gorgonia.Let(layer.Wih, Wih)
	gorgonia.Let(layer.Whh, Whh)
	gorgonia.Let(layer.Bih, Bih)
	gorgonia.Let(layer.Bhh, Bhh)
	return &layer
}

func (layer *Rnn) Forward(x *gorgonia.Node) *gorgonia.Node {
	if layer.Wih == nil {
		layer.Wih = gorgonia.NewTensor(x.Graph(), gorgonia.Float32, 3,
			gorgonia.WithShape(x.Shape()[0], layer.featureSize, layer.hidden),
			gorgonia.WithName("Wih"),
			gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	}
	if layer.Whh == nil {
		layer.Whh = gorgonia.NewTensor(x.Graph(), gorgonia.Float32, 3,
			gorgonia.WithShape(x.Shape()[0], layer.hidden, layer.hidden),
			gorgonia.WithName("Whh"),
			gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	}
	if layer.Bih == nil {
		layer.Bih = gorgonia.NewTensor(x.Graph(), gorgonia.Float32, 3,
			gorgonia.WithShape(x.Shape()[0], layer.steps, layer.hidden),
			gorgonia.WithName("Bih"),
			gorgonia.WithInit(gorgonia.Zeroes()))
	}
	if layer.Bhh == nil {
		layer.Bhh = gorgonia.NewTensor(x.Graph(), gorgonia.Float32, 3,
			gorgonia.WithShape(x.Shape()[0], layer.steps, layer.hidden),
			gorgonia.WithName("Bhh"),
			gorgonia.WithInit(gorgonia.Zeroes()))
	}
	var h *gorgonia.Node
	for i := 0; i < layer.steps; i++ {
		if i == 0 {
			h = gorgonia.Must(gorgonia.BatchedMatMul(x, layer.Wih))
			h = gorgonia.Must(gorgonia.Add(h, layer.Bih))
		} else {
			h = gorgonia.Must(gorgonia.BatchedMatMul(h, layer.Whh))
			h = gorgonia.Must(gorgonia.Add(h, layer.Bhh))
		}
		h = gorgonia.Must(gorgonia.Tanh(h))
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
