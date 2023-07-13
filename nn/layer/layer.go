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
	Args() map[string]float64
	Freeze()
	Unfreeze()
}

type base struct {
	init   initializer.Initializer
	name   string
	class  string
	device consts.DeviceType
}

type LayerCreateOption func(*base)

func WithInitializer(init initializer.Initializer) LayerCreateOption {
	return func(b *base) {
		b.init = init
	}
}

func WithDevice(device consts.DeviceType) LayerCreateOption {
	return func(b *base) {
		b.device = device
	}
}

func (b *base) new(class string, opts ...LayerCreateOption) {
	b.class = class
	b.device = consts.KCPU
	b.init = initializer.NewXavierUniform(1)
	for _, opt := range opts {
		opt(b)
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

func (b *base) Args() map[string]float64 {
	return nil
}

func (b *base) loadParam(data *pb.Dense) *tensor.Tensor {
	if data == nil {
		return nil
	}
	shape := make([]int64, 0, 2)
	for _, v := range data.Shape {
		shape = append(shape, int64(v))
	}
	t := tensor.FromFloat64(nil, data.GetData(),
		tensor.WithShapes(shape...),
		tensor.WithDevice(b.device))
	t.SetRequiresGrad(true)
	return t
}

func (b *base) initW(shapes ...int64) *tensor.Tensor {
	t := tensor.FromFloat64(nil, b.init.RandShape(shapes...),
		tensor.WithDevice(b.device),
		tensor.WithShapes(shapes...))
	t.SetRequiresGrad(true)
	return t
}

func (b *base) initB(shapes ...int64) *tensor.Tensor {
	t := tensor.Zeros(nil, consts.KDouble,
		tensor.WithDevice(b.device),
		tensor.WithShapes(shapes...))
	t.SetRequiresGrad(true)
	return t
}

func (b *base) Freeze() {
	panic("not implemented")
}

func (b *base) Unfreeze() {
	panic("not implemented")
}
