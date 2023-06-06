package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Lstm struct {
	*base
	featureSize, steps int
	hidden             int
	Wi, Bi             *gorgonia.Node
	Wf, Bf             *gorgonia.Node
	Wg, Bg             *gorgonia.Node
	Wo, Bo             *gorgonia.Node
}

func NewLstm(featureSize, steps, hidden int) *Lstm {
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
	layer.Wi = loadParam(g, params["Wi"], "Wi")
	layer.Wf = loadParam(g, params["Wf"], "Wf")
	layer.Wg = loadParam(g, params["Wg"], "Wg")
	layer.Wo = loadParam(g, params["Wo"], "Wo")
	layer.Bi = loadParam(g, params["Bi"], "Bi")
	layer.Bf = loadParam(g, params["Bf"], "Bf")
	layer.Bg = loadParam(g, params["Bg"], "Bg")
	layer.Bo = loadParam(g, params["Bo"], "Bo")
	return &layer
}

func (layer *Lstm) buildWeight(g *gorgonia.ExprGraph, name string) *gorgonia.Node {
	return gorgonia.NewMatrix(g, gorgonia.Float32,
		gorgonia.WithShape(layer.featureSize+layer.hidden, layer.hidden),
		gorgonia.WithName(name),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)))
}

func (layer *Lstm) buildBias(g *gorgonia.ExprGraph, name string, batchSize int) *gorgonia.Node {
	return gorgonia.NewMatrix(g, gorgonia.Float32,
		gorgonia.WithShape(batchSize, layer.hidden),
		gorgonia.WithName(name),
		gorgonia.WithInit(gorgonia.Zeroes()))
}

func (layer *Lstm) Forward(x, h, c *gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, *gorgonia.Node, error) {
	inputShape := x.Shape()
	if layer.Wi == nil {
		layer.Wi = layer.buildWeight(x.Graph(), "Wi")
	}
	if layer.Wf == nil {
		layer.Wf = layer.buildWeight(x.Graph(), "Wf")
	}
	if layer.Wg == nil {
		layer.Wg = layer.buildWeight(x.Graph(), "Wg")
	}
	if layer.Wo == nil {
		layer.Wo = layer.buildWeight(x.Graph(), "Wo")
	}
	if layer.Bi == nil {
		layer.Bi = layer.buildBias(x.Graph(), "Bi", inputShape[0])
	}
	if layer.Bf == nil {
		layer.Bf = layer.buildBias(x.Graph(), "Bf", inputShape[0])
	}
	if layer.Bg == nil {
		layer.Bg = layer.buildBias(x.Graph(), "Bg", inputShape[0])
	}
	if layer.Bo == nil {
		layer.Bo = layer.buildBias(x.Graph(), "Bo", inputShape[0])
	}
	if h == nil {
		h = gorgonia.NewMatrix(x.Graph(), gorgonia.Float32,
			gorgonia.WithShape(inputShape[0], layer.hidden), gorgonia.WithName("h"),
			gorgonia.WithInit(gorgonia.Zeroes()))
	}
	if c == nil {
		c = gorgonia.NewMatrix(x.Graph(), gorgonia.Float32,
			gorgonia.WithShape(inputShape[0], layer.hidden), gorgonia.WithName("c"),
			gorgonia.WithInit(gorgonia.Zeroes()))
	}
	x, err := gorgonia.Transpose(x, 1, 0, 2) // (steps, batch, feature)
	if err != nil {
		return nil, nil, nil, err
	}
	var result *gorgonia.Node
	for step := 0; step < layer.steps; step++ {
		t, err := gorgonia.Slice(x, gorgonia.S(step)) // (batch, feature)
		if err != nil {
			return nil, nil, nil, err
		}
		z := gorgonia.Must(gorgonia.Concat(1, h, t))    // (batch, feature+hidden)
		i := gorgonia.Must(gorgonia.Mul(z, layer.Wi))   // (batch, hidden)
		i = gorgonia.Must(gorgonia.Add(i, layer.Bi))    // (batch, hidden)
		i = gorgonia.Must(gorgonia.Sigmoid(i))          // (batch, hidden)
		f := gorgonia.Must(gorgonia.Mul(z, layer.Wf))   // (batch, hidden)
		f = gorgonia.Must(gorgonia.Add(f, layer.Bf))    // (batch, hidden)
		f = gorgonia.Must(gorgonia.Sigmoid(f))          // (batch, hidden)
		o := gorgonia.Must(gorgonia.Mul(z, layer.Wo))   // (batch, hidden)
		o = gorgonia.Must(gorgonia.Add(o, layer.Bo))    // (batch, hidden)
		o = gorgonia.Must(gorgonia.Sigmoid(o))          // (batch, hidden)
		g := gorgonia.Must(gorgonia.Mul(z, layer.Wg))   // (batch, hidden)
		g = gorgonia.Must(gorgonia.Add(g, layer.Bg))    // (batch, hidden)
		g = gorgonia.Must(gorgonia.Tanh(g))             // (batch, hidden)
		a := gorgonia.Must(gorgonia.HadamardProd(f, c)) // (batch, hidden)
		b := gorgonia.Must(gorgonia.HadamardProd(i, g)) // (batch, hidden)
		c = gorgonia.Must(gorgonia.Add(a, b))           // (batch, hidden)
		ct := gorgonia.Must(gorgonia.Tanh(c))           // (batch, hidden)
		h = gorgonia.Must(gorgonia.HadamardProd(o, ct)) // (batch, hidden)
		if result == nil {
			result = h
		} else {
			result = gorgonia.Must(gorgonia.Concat(0, result, h))
		}
	}
	return gorgonia.Must(gorgonia.Reshape(result,
		tensor.Shape{inputShape[0], inputShape[1], layer.hidden})), h, c, nil
}

func (layer *Lstm) Params() gorgonia.Nodes {
	return gorgonia.Nodes{
		layer.Wi, layer.Wf, layer.Wg, layer.Wo,
		layer.Bi, layer.Bf, layer.Bg, layer.Bo,
	}
}

func (layer *Lstm) Args() map[string]float32 {
	return map[string]float32{
		"feature_size": float32(layer.featureSize),
		"steps":        float32(layer.steps),
		"hidden":       float32(layer.hidden),
	}
}
