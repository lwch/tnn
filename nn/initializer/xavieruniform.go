package initializer

import (
	tensorInit "github.com/lwch/gotorch/init"
	"github.com/lwch/gotorch/tensor"
)

type XavierUniform struct {
	gain float64
}

var _ Initializer = &XavierUniform{}

func NewXavierUniform(gain float64) *XavierUniform {
	return &XavierUniform{
		gain: gain,
	}
}

func (rand *XavierUniform) Init(t *tensor.Tensor) {
	tensorInit.XaiverUniform(t, rand.gain)
}
