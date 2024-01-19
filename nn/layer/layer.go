package layer

import (
	"errors"
	"fmt"
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
	init      initializer.Initializer
	name      string
	class     string
	device    consts.DeviceType
	paramType consts.ScalarType
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

func WithParamType(t consts.ScalarType) LayerCreateOption {
	return func(b *base) {
		b.paramType = t
	}
}

func (b *base) new(class, name string, opts ...LayerCreateOption) {
	b.class = class
	b.name = name
	b.device = consts.KCPU
	b.init = initializer.NewXavierUniform(1)
	b.paramType = consts.KFloat
	for _, opt := range opts {
		opt(b)
	}
	switch b.paramType {
	case consts.KBFloat16:
	case consts.KHalf:
	case consts.KFloat:
	case consts.KDouble:
		// permit
	default:
		panic(fmt.Errorf("unsupported param type: %s", b.paramType.String()))
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
	t := tensor.Zeros(b.paramType,
		tensor.WithDevice(b.device),
		tensor.WithShapes(shapes...))
	b.init.Init(t)
	t.SetRequiresGrad(true)
	return t
}

func randB[T float32 | float64](shapes ...int64) []T {
	n := shapes[0]
	for i := 1; i < len(shapes); i++ {
		n *= shapes[i]
	}
	data := make([]T, n)
	for i := 0; i < len(data); i++ {
		data[i] = T(rand.NormFloat64())
	}
	return data
}

func (b *base) initB(shapes ...int64) *tensor.Tensor {
	opts := []tensor.Option{
		tensor.WithDevice(b.device),
		tensor.WithShapes(shapes...),
	}
	var t *tensor.Tensor
	switch b.paramType {
	case consts.KBFloat16:
		data := randB[float32](shapes...)
		t = tensor.FromBFloat16(data, opts...)
	case consts.KHalf:
		data := randB[float32](shapes...)
		t = tensor.FromHalf(data, opts...)
	case consts.KFloat:
		data := randB[float32](shapes...)
		t = tensor.FromFloat32(data, opts...)
	case consts.KDouble:
		data := randB[float64](shapes...)
		t = tensor.FromFloat64(data, opts...)
	}
	t.SetRequiresGrad(true)
	return t
}

func (b *base) initN(n float64) *tensor.Tensor {
	switch b.paramType {
	case consts.KBFloat16:
		return tensor.FromBFloat16([]float32{float32(n)},
			tensor.WithShapes(1),
			tensor.WithDevice(b.device))
	case consts.KHalf:
		return tensor.FromHalf([]float32{float32(n)},
			tensor.WithShapes(1),
			tensor.WithDevice(b.device))
	case consts.KFloat:
		return tensor.FromFloat32([]float32{float32(n)},
			tensor.WithShapes(1),
			tensor.WithDevice(b.device))
	case consts.KDouble:
		return tensor.FromFloat64([]float64{float64(n)},
			tensor.WithShapes(1),
			tensor.WithDevice(b.device))
	default:
		panic(errors.New("can not reach here"))
	}
}

func (b *base) ones(shapes ...int64) *tensor.Tensor {
	n := shapes[0]
	for i := 1; i < len(shapes); i++ {
		n *= shapes[i]
	}
	data := make([]float32, n)
	for i := int64(0); i < n; i++ {
		data[i] = 1
	}
	var fn func([]float32, ...tensor.Option) *tensor.Tensor
	switch b.paramType {
	case consts.KBFloat16:
		fn = tensor.FromBFloat16
	case consts.KHalf:
		fn = tensor.FromHalf
	case consts.KFloat:
		fn = tensor.FromFloat32
	case consts.KDouble:
		data := make([]float64, n)
		for i := int64(0); i < n; i++ {
			data[i] = 1
		}
		return tensor.FromFloat64(data,
			tensor.WithShapes(shapes...),
			tensor.WithDevice(b.device))
	}
	return fn(data,
		tensor.WithShapes(shapes...),
		tensor.WithDevice(b.device))
}

func (b *base) Freeze() {
	panic("not implemented")
}

func (b *base) Unfreeze() {
	panic("not implemented")
}
