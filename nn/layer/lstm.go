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
	Wi, Bi             tensor.Tensor
	Wf, Bf             tensor.Tensor
	Wg, Bg             tensor.Tensor
	Wo, Bo             tensor.Tensor
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
	layer.Wi = loadParam(g, params["Wi"])
	layer.Wf = loadParam(g, params["Wf"])
	layer.Wg = loadParam(g, params["Wg"])
	layer.Wo = loadParam(g, params["Wo"])
	layer.Bi = loadParam(g, params["Bi"])
	layer.Bf = loadParam(g, params["Bf"])
	layer.Bg = loadParam(g, params["Bg"])
	layer.Bo = loadParam(g, params["Bo"])
	return &layer
}

func (layer *Lstm) Forward(x, h, c *gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, *gorgonia.Node, error) {
	inputShape := x.Shape()
	if layer.Wi == nil {
		layer.Wi = initW(layer.featureSize+layer.hidden, layer.hidden)
	}
	if layer.Wf == nil {
		layer.Wf = initW(layer.featureSize+layer.hidden, layer.hidden)
	}
	if layer.Wg == nil {
		layer.Wg = initW(layer.featureSize+layer.hidden, layer.hidden)
	}
	if layer.Wo == nil {
		layer.Wo = initW(layer.featureSize+layer.hidden, layer.hidden)
	}
	if layer.Bi == nil {
		layer.Bi = initB(inputShape[0], layer.hidden)
	}
	if layer.Bf == nil {
		layer.Bf = initB(inputShape[0], layer.hidden)
	}
	if layer.Bg == nil {
		layer.Bg = initB(inputShape[0], layer.hidden)
	}
	if layer.Bo == nil {
		layer.Bo = initB(inputShape[0], layer.hidden)
	}
	Wi := gorgonia.NodeFromAny(x.Graph(), layer.Wi, gorgonia.WithName("Wi"))
	Wf := gorgonia.NodeFromAny(x.Graph(), layer.Wf, gorgonia.WithName("Wf"))
	Wg := gorgonia.NodeFromAny(x.Graph(), layer.Wg, gorgonia.WithName("Wg"))
	Wo := gorgonia.NodeFromAny(x.Graph(), layer.Wo, gorgonia.WithName("Wo"))
	Bi := gorgonia.NodeFromAny(x.Graph(), layer.Bi, gorgonia.WithName("Bi"))
	Bf := gorgonia.NodeFromAny(x.Graph(), layer.Bf, gorgonia.WithName("Bf"))
	Bg := gorgonia.NodeFromAny(x.Graph(), layer.Bg, gorgonia.WithName("Bg"))
	Bo := gorgonia.NodeFromAny(x.Graph(), layer.Bo, gorgonia.WithName("Bo"))
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
		i := gorgonia.Must(gorgonia.Mul(z, Wi))         // (batch, hidden)
		i = gorgonia.Must(gorgonia.Add(i, Bi))          // (batch, hidden)
		i = gorgonia.Must(gorgonia.Sigmoid(i))          // (batch, hidden)
		f := gorgonia.Must(gorgonia.Mul(z, Wf))         // (batch, hidden)
		f = gorgonia.Must(gorgonia.Add(f, Bf))          // (batch, hidden)
		f = gorgonia.Must(gorgonia.Sigmoid(f))          // (batch, hidden)
		o := gorgonia.Must(gorgonia.Mul(z, Wo))         // (batch, hidden)
		o = gorgonia.Must(gorgonia.Add(o, Bo))          // (batch, hidden)
		o = gorgonia.Must(gorgonia.Sigmoid(o))          // (batch, hidden)
		g := gorgonia.Must(gorgonia.Mul(z, Wg))         // (batch, hidden)
		g = gorgonia.Must(gorgonia.Add(g, Bg))          // (batch, hidden)
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

func (layer *Lstm) Params() map[string]tensor.Tensor {
	return map[string]tensor.Tensor{
		"Wi": layer.Wi,
		"Wf": layer.Wf,
		"Wg": layer.Wg,
		"Wo": layer.Wo,
		"Bi": layer.Bi,
		"Bf": layer.Bf,
		"Bg": layer.Bg,
		"Bo": layer.Bo,
	}
}

func (layer *Lstm) Args() map[string]float32 {
	return map[string]float32{
		"feature_size": float32(layer.featureSize),
		"steps":        float32(layer.steps),
		"hidden":       float32(layer.hidden),
	}
}
