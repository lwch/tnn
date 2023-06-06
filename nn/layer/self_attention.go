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
	scale       tensor.Tensor
	// params
	wq, wk, wv tensor.Tensor
	bq, bk, bv tensor.Tensor
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
	layer.base = new("self_attention")
	layer.name = name
	layer.steps = int(args["steps"])
	layer.dims = int(args["dims"])
	layer.wq = loadParam(g, params["Wq"])
	layer.wk = loadParam(g, params["Wk"])
	layer.wv = loadParam(g, params["Wv"])
	layer.bq = loadParam(g, params["Bq"])
	layer.bk = loadParam(g, params["Bk"])
	layer.bv = loadParam(g, params["Bv"])
	return &layer
}

func (layer *SelfAttention) Forward(x *gorgonia.Node) (*gorgonia.Node, gorgonia.Nodes) {
	inputShape := x.Shape()
	if layer.scale == nil {
		layer.scale = tensor.New(tensor.FromScalar(
			float32(math.Sqrt(float64(layer.dims)))))
	}
	if layer.wq == nil {
		layer.wq = initW(layer.dims, layer.dims)
	}
	if layer.wk == nil {
		layer.wk = initW(layer.dims, layer.dims)
	}
	if layer.wv == nil {
		layer.wv = initW(layer.dims, layer.dims)
	}
	if layer.bq == nil {
		layer.bq = initB(layer.steps, layer.dims)
	}
	if layer.bk == nil {
		layer.bk = initB(layer.steps, layer.dims)
	}
	if layer.bv == nil {
		layer.bv = initB(layer.steps, layer.dims)
	}
	Wq := gorgonia.NodeFromAny(x.Graph(), layer.wq, gorgonia.WithName("Wq"))
	Wk := gorgonia.NodeFromAny(x.Graph(), layer.wk, gorgonia.WithName("Wk"))
	Wv := gorgonia.NodeFromAny(x.Graph(), layer.wv, gorgonia.WithName("Wv"))
	Bq := gorgonia.NodeFromAny(x.Graph(), layer.bq, gorgonia.WithName("Bq"))
	Bk := gorgonia.NodeFromAny(x.Graph(), layer.bk, gorgonia.WithName("Bk"))
	Bv := gorgonia.NodeFromAny(x.Graph(), layer.bv, gorgonia.WithName("Bv"))
	scale := gorgonia.NodeFromAny(x.Graph(), layer.scale, gorgonia.WithName("scale"))
	var result *gorgonia.Node
	for batch := 0; batch < inputShape[0]; batch++ {
		x := gorgonia.Must(gorgonia.Slice(x, gorgonia.S(batch)))
		q := gorgonia.Must(gorgonia.Mul(x, Wq))
		q = gorgonia.Must(gorgonia.Add(q, Bq))
		k := gorgonia.Must(gorgonia.Mul(x, Wk))
		k = gorgonia.Must(gorgonia.Add(k, Bk))
		v := gorgonia.Must(gorgonia.Mul(x, Wv))
		v = gorgonia.Must(gorgonia.Add(v, Bv))
		k = gorgonia.Must(gorgonia.Transpose(k))
		a := gorgonia.Must(gorgonia.Mul(q, k))
		a = gorgonia.Must(gorgonia.Div(a, scale))
		a = gorgonia.Must(gorgonia.SoftMax(a, 1))
		a = gorgonia.Must(gorgonia.Mul(a, v))
		if result == nil {
			result = a
		} else {
			result = gorgonia.Must(gorgonia.Concat(0, result, a))
		}
	}
	return gorgonia.Must(gorgonia.Reshape(result, tensor.Shape{inputShape[0], layer.steps, layer.dims})),
		gorgonia.Nodes{Wq, Wk, Wv, Bq, Bk, Bv}
}

func (layer *SelfAttention) Params() map[string]tensor.Tensor {
	return map[string]tensor.Tensor{
		"Wq": layer.wq, "Wk": layer.wk, "Wv": layer.wv,
		"Bq": layer.bq, "Bk": layer.bk, "Bv": layer.bv,
	}
}

func (layer *SelfAttention) Args() map[string]float32 {
	return map[string]float32{
		"steps": float32(layer.steps),
		"dims":  float32(layer.dims),
	}
}
