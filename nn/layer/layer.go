package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Layer interface {
	Params() map[string]tensor.Tensor
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

func (b *base) Params() map[string]tensor.Tensor {
	return nil
}

func (b *base) Args() map[string]float32 {
	return nil
}

func loadParam(g *gorgonia.ExprGraph, data *pb.Dense) tensor.Tensor {
	if data == nil {
		return nil
	}
	shape := make([]int, 0, 2)
	for _, v := range data.Shape {
		shape = append(shape, int(v))
	}
	return tensor.New(tensor.WithShape(shape...), tensor.WithBacking(data.GetData()))
}

func initW(shapes ...int) tensor.Tensor {
	return tensor.New(
		tensor.WithShape(shapes...),
		tensor.WithBacking(gorgonia.GlorotEtAlU32(1.0, shapes...)))
}

func initB(shapes ...int) tensor.Tensor {
	total := 1
	for _, v := range shapes {
		total *= v
	}
	return tensor.New(
		tensor.WithShape(shapes...),
		tensor.WithBacking(make([]float32, total)))
}
