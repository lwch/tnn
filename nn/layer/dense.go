package layer

import (
	"gorgonia.org/gorgonia"
)

type Dense struct {
	w *gorgonia.Node
	b *gorgonia.Node
}

func NewDense(g *gorgonia.ExprGraph, batchSize, input, output int) Layer {
	return &Dense{
		w: gorgonia.NewMatrix(g, gorgonia.Float32,
			gorgonia.WithShape(input, output),
			gorgonia.WithName("w"),
			gorgonia.WithInit(gorgonia.GlorotN(1.0))),
		b: gorgonia.NewMatrix(g, gorgonia.Float32,
			gorgonia.WithShape(batchSize, output),
			gorgonia.WithName("b"),
			gorgonia.WithInit(gorgonia.Zeroes())),
	}
}

func (layer *Dense) Forward(x *gorgonia.Node) *gorgonia.Node {
	wx := gorgonia.Must(gorgonia.Mul(x, layer.w))
	return gorgonia.Must(gorgonia.Add(wx, layer.b))
}

func (layer *Dense) Params() gorgonia.Nodes {
	return gorgonia.Nodes{layer.w, layer.b}
}
