package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Layer interface {
	Forward(x *gorgonia.Node) *gorgonia.Node
	Params() gorgonia.Nodes
	Class() string
	SetName(name string)
	Name() string
	Args() map[string]float32
}

type base struct {
	name  string
	class string
}

func new(class string) *base {
	return &base{
		class: class,
	}
}

func (b *base) Class() string {
	return b.class
}

func (b *base) SetName(name string) {
	b.name = name
}

func (b *base) Name() string {
	return b.name
}

func (b *base) Params() gorgonia.Nodes {
	return nil
}

func (b *base) Args() map[string]float32 {
	return nil
}

func loadParam(g *gorgonia.ExprGraph, data *pb.Dense, name string) *gorgonia.Node {
	if data == nil {
		return nil
	}
	shape := make([]int, 0, 2)
	for _, v := range data.Shape {
		shape = append(shape, int(v))
	}
	value := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(data.GetData()))
	ret := gorgonia.NewTensor(g, gorgonia.Float32, len(shape),
		gorgonia.WithShape(shape...),
		gorgonia.WithName(name))
	gorgonia.Let(ret, value)
	return ret
}
