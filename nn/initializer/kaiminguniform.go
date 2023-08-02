package initializer

import (
	tensorInit "github.com/lwch/gotorch/init"
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
	tensorInit.KaimingUniform(t, rand.a)
}
