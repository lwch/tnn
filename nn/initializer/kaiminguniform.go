package initializer

import (
	"github.com/lwch/gotorch/tensor"
)

type KaimingUniform struct {
	a float64
}

var _ Initializer = &KaimingUniform{}

func NewKaimingUniform(a float64) *KaimingUniform {
	return &KaimingUniform{
		a: a,
	}
}

func (rand *KaimingUniform) Init(t *tensor.Tensor) {
	tensor.KaimingUniform(t, rand.a)
}
