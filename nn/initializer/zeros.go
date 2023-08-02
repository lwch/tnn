package initializer

import (
	tensorInit "github.com/lwch/gotorch/init"
	"github.com/lwch/gotorch/tensor"
)

type Zeros struct {
}

var _ Initializer = &Zeros{}

func NewZeros() *Zeros {
	return &Zeros{}
}

func (*Zeros) Init(t *tensor.Tensor) {
	tensorInit.Zeros(t)
}
