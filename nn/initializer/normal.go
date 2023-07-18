package initializer

import (
	"time"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
)

type Normal struct {
	n distuv.Normal
}

var _ Initializer = &Normal{}

func NewNormal(mean, stddev float64) *Normal {
	return &Normal{
		n: distuv.Normal{
			Mu:    mean,
			Sigma: stddev,
			Src:   rand.NewSource(uint64(time.Now().UnixNano())),
		},
	}
}

func (rand *Normal) Rand() float32 {
	return float32(rand.n.Rand())
}

func (rand *Normal) RandN(n int) []float32 {
	ret := make([]float32, n)
	for i := 0; i < n; i++ {
		ret[i] = float32(rand.n.Rand())
	}
	return ret
}

func (rand *Normal) RandShape(shapes ...int64) []float32 {
	size := int64(1)
	for _, s := range shapes {
		size *= s
	}
	ret := make([]float32, size)
	for i := 0; i < int(size); i++ {
		ret[i] = float32(rand.n.Rand())
	}
	return ret
}
