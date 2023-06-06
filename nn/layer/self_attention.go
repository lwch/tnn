package layer

import (
	"math"

	"github.com/lwch/tnn/internal/pb"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type SelfAttention struct {
	*base
	steps, dims int
	scale       *gorgonia.Node
	// params
	wq, wk, wv *gorgonia.Node
	bq, bk, bv *gorgonia.Node
}

func NewSelfAttention(steps, dims int) *SelfAttention {
	var layer SelfAttention
	layer.base = new("self_attention")
	layer.steps = steps
	layer.dims = dims
	return &layer
}

func LoadSelfAttention(g *gorgonia.ExprGraph, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer SelfAttention
	layer.base = new("dense")
	layer.name = name
	layer.steps = int(args["steps"])
	layer.dims = int(args["dims"])
	layer.wq = loadParam(g, params["Wq"], "Wq")
	layer.wk = loadParam(g, params["Wk"], "Wk")
	layer.wv = loadParam(g, params["Wv"], "Wv")
	layer.bq = loadParam(g, params["bq"], "bq")
	layer.bk = loadParam(g, params["bk"], "bk")
	layer.bv = loadParam(g, params["bv"], "bv")
	return &layer
}

func (layer *SelfAttention) Forward(x *gorgonia.Node) *gorgonia.Node {
	inputShape := x.Shape()
	if layer.scale == nil {
		layer.scale = gorgonia.NewScalar(x.Graph(), gorgonia.Float32,
			gorgonia.WithName("scale"),
			gorgonia.WithValue(float32(math.Sqrt(float64(layer.dims)))))
	}
	if layer.wq == nil {
		layer.wq = gorgonia.NewMatrix(x.Graph(), gorgonia.Float32,
			gorgonia.WithShape(layer.dims, layer.dims),
			gorgonia.WithName("Wq"),
			gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	}
	if layer.wk == nil {
		layer.wk = gorgonia.NewMatrix(x.Graph(), gorgonia.Float32,
			gorgonia.WithShape(layer.dims, layer.dims),
			gorgonia.WithName("Wk"),
			gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	}
	if layer.wv == nil {
		layer.wv = gorgonia.NewMatrix(x.Graph(), gorgonia.Float32,
			gorgonia.WithShape(layer.dims, layer.dims),
			gorgonia.WithName("Wv"),
			gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	}
	if layer.bq == nil {
		layer.bq = gorgonia.NewMatrix(x.Graph(), gorgonia.Float32,
			gorgonia.WithShape(layer.steps, layer.dims),
			gorgonia.WithName("bq"),
			gorgonia.WithInit(gorgonia.Zeroes()))
	}
	if layer.bk == nil {
		layer.bk = gorgonia.NewMatrix(x.Graph(), gorgonia.Float32,
			gorgonia.WithShape(layer.steps, layer.dims),
			gorgonia.WithName("bk"),
			gorgonia.WithInit(gorgonia.Zeroes()))
	}
	if layer.bv == nil {
		layer.bv = gorgonia.NewMatrix(x.Graph(), gorgonia.Float32,
			gorgonia.WithShape(layer.steps, layer.dims),
			gorgonia.WithName("bv"),
			gorgonia.WithInit(gorgonia.Zeroes()))
	}
	var result *gorgonia.Node
	for batch := 0; batch < inputShape[0]; batch++ {
		x := gorgonia.Must(gorgonia.Slice(x, gorgonia.S(batch)))
		q := gorgonia.Must(gorgonia.Mul(x, layer.wq))
		q = gorgonia.Must(gorgonia.Add(q, layer.bq))
		k := gorgonia.Must(gorgonia.Mul(x, layer.wk))
		k = gorgonia.Must(gorgonia.Add(k, layer.bk))
		v := gorgonia.Must(gorgonia.Mul(x, layer.wv))
		v = gorgonia.Must(gorgonia.Add(v, layer.bv))
		k = gorgonia.Must(gorgonia.Transpose(k))
		a := gorgonia.Must(gorgonia.Mul(q, k))
		a = gorgonia.Must(gorgonia.Div(a, layer.scale))
		a = gorgonia.Must(gorgonia.SoftMax(a, 1))
		a = gorgonia.Must(gorgonia.Mul(a, v))
		if result == nil {
			result = a
		} else {
			result = gorgonia.Must(gorgonia.Concat(0, result, a))
		}
	}
	return gorgonia.Must(gorgonia.Reshape(result, tensor.Shape{inputShape[0], layer.steps, layer.dims}))
}

func (layer *SelfAttention) Params() gorgonia.Nodes {
	return gorgonia.Nodes{
		layer.wq, layer.wk, layer.wv,
		layer.bq, layer.bk, layer.bv,
	}
}

func (layer *SelfAttention) Args() map[string]float32 {
	return map[string]float32{
		"steps": float32(layer.steps),
		"dims":  float32(layer.dims),
	}
}
