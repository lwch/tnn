package initializer

import (
	tensorInit "github.com/lwch/gotorch/init"
	"github.com/lwch/gotorch/tensor"
)

type Normal struct {
	mean  float64
	stdev float64
}

var _ Initializer = &Normal{}

func NewNormal(mean, stddev float64) *Normal {
	return &Normal{
		mean:  mean,
		stdev: stddev,
	}
}

func (rand *Normal) Init(t *tensor.Tensor) {
	tensorInit.Normal(t, rand.mean, rand.stdev)
}
