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
	Name() string
	Args() map[string]float32
}

type base struct {
	name  string
	class string
}

func new(class, name string) *base {
	return &base{
		class: class,
		name:  name,
	}
}

func (b *base) Class() string {
	return b.class
}

func (b *base) Name() string {
	return b.name
}

func (b *base) Args() map[string]float32 {
	return nil
}

func loadParam(data *pb.Dense) tensor.Tensor {
	if data == nil {
		return nil
	}
	shape := make([]int, 0, 2)
	for _, v := range data.Shape {
		shape = append(shape, int(v))
	}
	return tensor.New(tensor.WithShape(shape...), tensor.WithBacking(data.GetData()))
}
