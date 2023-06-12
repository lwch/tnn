package activation

import "github.com/lwch/gotorch/tensor"

type base struct {
	class string
	name  string
}

func new(class string) *base {
	return &base{
		class: class,
	}
}

func (layer *base) Class() string {
	return layer.class
}

func (layer *base) SetName(name string) {
	layer.name = name
}

func (layer *base) Name() string {
	if len(layer.name) == 0 {
		return layer.class
	}
	return layer.name
}

func (layer *base) Params() map[string]*tensor.Tensor {
	return nil
}

func (*base) Args() map[string]float32 {
	return nil
}
