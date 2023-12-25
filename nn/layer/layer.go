package layer

import (
	"math/rand"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/initializer"
)

type Layer interface {
	Params() map[string]*tensor.Tensor
	Class() string
	Name() string
	Args() map[string]float32
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

func (b *base) new(class, name string, opts ...LayerCreateOption) {
	b.class = class
	b.name = name
	b.device = consts.KCPU
	b.init = initializer.NewXavierUniform(1)
	for _, opt := range opts {
		opt(b)
	}
}

func (b *base) Class() string {
	return b.class
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

func (b *base) initW(shapes ...int64) *tensor.Tensor {
	t := tensor.Zeros(consts.KFloat,
		tensor.WithDevice(b.device),
		tensor.WithShapes(shapes...))
	b.init.Init(t)
	t.SetRequiresGrad(true)
	return t
}

func (b *base) initB(shapes ...int64) *tensor.Tensor {
	n := shapes[0]
	for i := 1; i < len(shapes); i++ {
		n *= shapes[i]
	}
	data := make([]float32, n)
	for i := 0; i < len(data); i++ {
		data[i] = float32(rand.NormFloat64())
	}
	t := tensor.FromFloat32(data,
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
