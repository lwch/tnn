package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"gorgonia.org/gorgonia"
)

type Lstm struct {
	*base
	featureSize, steps int
	hidden             int
	// it
	Wii *gorgonia.Node
	Whi *gorgonia.Node
	Bii *gorgonia.Node
	Bhi *gorgonia.Node
	// ft
	Wif *gorgonia.Node
	Whf *gorgonia.Node
	Bif *gorgonia.Node
	Bhf *gorgonia.Node
	// gt
	Wig *gorgonia.Node
	Whg *gorgonia.Node
	Big *gorgonia.Node
	Bhg *gorgonia.Node
	// ot
	Wio *gorgonia.Node
	Who *gorgonia.Node
	Bio *gorgonia.Node
	Bho *gorgonia.Node
}

func NewLstm(g *gorgonia.ExprGraph, featureSize, steps, hidden int) Layer {
	var layer Lstm
	layer.base = new("lstm")
	layer.featureSize = featureSize
	layer.steps = steps
	layer.hidden = hidden
	return &layer
}

func LoadLstm(g *gorgonia.ExprGraph, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Lstm
	layer.base = new("lstm")
	layer.name = name
	layer.featureSize = int(args["feature_size"])
	layer.steps = int(args["steps"])
	layer.hidden = int(args["hidden"])
	// it
	layer.Wii = loadParam(g, params["Wii"], "Wii")
	layer.Whi = loadParam(g, params["Whi"], "Whi")
	layer.Bii = loadParam(g, params["Bii"], "Bii")
	layer.Bhi = loadParam(g, params["Bhi"], "Bhi")
	// ft
	layer.Wif = loadParam(g, params["Wif"], "Wif")
	layer.Whf = loadParam(g, params["Whf"], "Whf")
	layer.Bif = loadParam(g, params["Bif"], "Bif")
	layer.Bhf = loadParam(g, params["Bhf"], "Bhf")
	// gt
	layer.Wig = loadParam(g, params["Wig"], "Wig")
	layer.Whg = loadParam(g, params["Whg"], "Whg")
	layer.Big = loadParam(g, params["Big"], "Big")
	layer.Bhg = loadParam(g, params["Bhg"], "Bhg")
	// ot
	layer.Wio = loadParam(g, params["Wio"], "Wio")
	layer.Who = loadParam(g, params["Who"], "Who")
	layer.Bio = loadParam(g, params["Bio"], "Bio")
	layer.Bho = loadParam(g, params["Bho"], "Bho")
	return &layer
}

func (layer *Lstm) Forward(x *gorgonia.Node, _ bool) *gorgonia.Node {
	// it
	blockIt := buildRnnBlock(x, []*gorgonia.Node{layer.Wii, layer.Whi, layer.Bii, layer.Bhi},
		[]string{"Wii", "Whi", "Bii", "Bhi"}, layer.featureSize, layer.steps, layer.hidden)
	layer.Wii = blockIt[0]
	layer.Whi = blockIt[1]
	layer.Bii = blockIt[2]
	layer.Bhi = blockIt[3]
	// ft
	blockFt := buildRnnBlock(x, []*gorgonia.Node{layer.Wif, layer.Whf, layer.Bif, layer.Bhf},
		[]string{"Wif", "Whf", "Bif", "Bhf"}, layer.featureSize, layer.steps, layer.hidden)
	layer.Wif = blockFt[0]
	layer.Whf = blockFt[1]
	layer.Bif = blockFt[2]
	layer.Bhf = blockFt[3]
	// gt
	blockGt := buildRnnBlock(x, []*gorgonia.Node{layer.Wig, layer.Whg, layer.Big, layer.Bhg},
		[]string{"Wig", "Whg", "Big", "Bhg"}, layer.featureSize, layer.steps, layer.hidden)
	layer.Wig = blockGt[0]
	layer.Whg = blockGt[1]
	layer.Big = blockGt[2]
	layer.Bhg = blockGt[3]
	// ot
	blockOt := buildRnnBlock(x, []*gorgonia.Node{layer.Wio, layer.Who, layer.Bio, layer.Bho},
		[]string{"Wio", "Who", "Bio", "Bho"}, layer.featureSize, layer.steps, layer.hidden)
	layer.Wio = blockOt[0]
	layer.Who = blockOt[1]
	layer.Bio = blockOt[2]
	layer.Bho = blockOt[3]
	h := gorgonia.NewTensor(x.Graph(), gorgonia.Float32, 3,
		gorgonia.WithShape(x.Shape()[0], layer.steps, layer.hidden), gorgonia.WithName("h"),
		gorgonia.WithInit(gorgonia.Zeroes()))
	c := gorgonia.NewTensor(x.Graph(), gorgonia.Float32, 3,
		gorgonia.WithShape(x.Shape()[0], layer.steps, layer.hidden), gorgonia.WithName("c"),
		gorgonia.WithInit(gorgonia.Zeroes()))
	for i := 0; i < layer.steps; i++ {
		a := gorgonia.Must(gorgonia.BatchedMatMul(x, layer.Wii))
		a = gorgonia.Must(gorgonia.Add(a, layer.Bii))
		b := gorgonia.Must(gorgonia.BatchedMatMul(h, layer.Whi))
		b = gorgonia.Must(gorgonia.Add(b, layer.Bhi))
		it := gorgonia.Must(gorgonia.Sigmoid(gorgonia.Must(gorgonia.Add(a, b))))
		a = gorgonia.Must(gorgonia.BatchedMatMul(x, layer.Wif))
		a = gorgonia.Must(gorgonia.Add(a, layer.Bif))
		b = gorgonia.Must(gorgonia.BatchedMatMul(h, layer.Whf))
		b = gorgonia.Must(gorgonia.Add(b, layer.Bhf))
		ft := gorgonia.Must(gorgonia.Sigmoid(gorgonia.Must(gorgonia.Add(a, b))))
		a = gorgonia.Must(gorgonia.BatchedMatMul(x, layer.Wig))
		a = gorgonia.Must(gorgonia.Add(a, layer.Big))
		b = gorgonia.Must(gorgonia.BatchedMatMul(h, layer.Whg))
		b = gorgonia.Must(gorgonia.Add(b, layer.Bhg))
		gt := gorgonia.Must(gorgonia.Tanh(gorgonia.Must(gorgonia.Add(a, b))))
		a = gorgonia.Must(gorgonia.BatchedMatMul(x, layer.Wio))
		a = gorgonia.Must(gorgonia.Add(a, layer.Bio))
		b = gorgonia.Must(gorgonia.BatchedMatMul(h, layer.Who))
		b = gorgonia.Must(gorgonia.Add(b, layer.Bho))
		ot := gorgonia.Must(gorgonia.Sigmoid(gorgonia.Must(gorgonia.Add(a, b))))
		a = gorgonia.Must(gorgonia.HadamardProd(ft, c))
		b = gorgonia.Must(gorgonia.HadamardProd(it, gt))
		c = gorgonia.Must(gorgonia.Add(a, b))
		h = gorgonia.Must(gorgonia.HadamardProd(ot, gorgonia.Must(gorgonia.Tanh(c))))
	}
	return h
}

func (layer *Lstm) Params() gorgonia.Nodes {
	return gorgonia.Nodes{
		layer.Wii, layer.Whi, layer.Bii, layer.Bhi,
		layer.Wif, layer.Whf, layer.Bif, layer.Bhf,
		layer.Wig, layer.Whg, layer.Big, layer.Bhg,
		layer.Wio, layer.Who, layer.Bio, layer.Bho,
	}
}

func (layer *Lstm) Args() map[string]float32 {
	return map[string]float32{
		"feature_size": float32(layer.featureSize),
		"steps":        float32(layer.steps),
		"hidden":       float32(layer.hidden),
	}
}
