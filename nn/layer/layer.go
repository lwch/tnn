package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/initializer"
)

type Layer interface {
	Params() map[string]*tensor.Tensor
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

func (b *base) Params() map[string]*tensor.Tensor {
	return nil
}

func (b *base) Args() map[string]float32 {
	return nil
}

func loadParam(data *pb.Dense) *tensor.Tensor {
	if data == nil {
		return nil
	}
	shape := make([]int64, 0, 2)
	for _, v := range data.Shape {
		shape = append(shape, int64(v))
	}
	t := tensor.FromFloat32(nil, data.GetData(), shape...)
	t.SetRequiresGrad(true)
	return t
}

var wInitializer = initializer.NewXavierUniform(1)

func initW(shapes ...int64) *tensor.Tensor {
	t := tensor.FromFloat32(nil, wInitializer.RandShape(shapes...), shapes...)
	t.SetRequiresGrad(true)
	return t
}

func initB(shapes ...int64) *tensor.Tensor {
	t := tensor.Zeros(nil, consts.KFloat, shapes...)
	t.SetRequiresGrad(true)
	return t
}
