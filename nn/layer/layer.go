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
	Freeze()
	Unfreeze()
}

type base struct {
	name   string
	class  string
	device consts.DeviceType
}

func new(class string, device consts.DeviceType) *base {
	return &base{
		class:  class,
		device: device,
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

func (b *base) loadParam(data *pb.Dense) *tensor.Tensor {
	if data == nil {
		return nil
	}
	shape := make([]int64, 0, 2)
	for _, v := range data.Shape {
		shape = append(shape, int64(v))
	}
	t := tensor.FromFloat32(nil, data.GetData(),
		tensor.WithShapes(shape...),
		tensor.WithDevice(b.device))
	t.SetRequiresGrad(true)
	return t
}

var wInitializer = initializer.NewNormal(0, 0.001)

func (b *base) initW(shapes ...int64) *tensor.Tensor {
	t := tensor.FromFloat32(nil, wInitializer.RandShape(shapes...),
		tensor.WithDevice(b.device),
		tensor.WithShapes(shapes...))
	t.SetRequiresGrad(true)
	return t
}

var bInitializer = initializer.NewNormal(0, 0.001)

func (b *base) initB(shapes ...int64) *tensor.Tensor {
	t := tensor.FromFloat32(nil, bInitializer.RandShape(shapes...),
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
