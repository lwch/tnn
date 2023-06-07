package layer

import (
	"github.com/lwch/runtime"
	"github.com/lwch/tnn/internal/pb"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

type Layer interface {
	Params() map[string]*ts.Tensor
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

func (b *base) Params() map[string]*ts.Tensor {
	return nil
}

func (b *base) Args() map[string]float32 {
	return nil
}

func loadParam(data *pb.Dense) *ts.Tensor {
	if data == nil {
		return nil
	}
	shape := make([]int64, 0, 2)
	for _, v := range data.Shape {
		shape = append(shape, int64(v))
	}
	t, err := ts.NewTensorFromData(data.GetData(), shape)
	runtime.Assert(err)
	return t
}

func initW(vs *nn.Path, name string, dims ...int64) *ts.Tensor {
	return vs.MustKaimingUniform(name, dims)
}

func initB(vs *nn.Path, name string, dims ...int64) *ts.Tensor {
	return vs.MustZeros(name, dims)
}
