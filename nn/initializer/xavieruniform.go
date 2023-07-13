package initializer

import (
	"math"
	"time"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
)

type XavierUniform struct {
	n    distuv.Uniform
	gain float64
}

var _ Initializer = &XavierUniform{}

func NewXavierUniform(gain float64) *XavierUniform {
	return &XavierUniform{
		n: distuv.Uniform{
			Min: 0,
			Max: 1,
			Src: rand.NewSource(uint64(time.Now().UnixNano())),
		},
		gain: gain,
	}
}

func (rand *XavierUniform) Rand() float64 {
	a := rand.gain * math.Sqrt(6/(rand.n.Min+rand.n.Max))
	rand.n.Min = -a
	rand.n.Max = a
	return rand.n.Rand()
}

func (rand *XavierUniform) RandN(n int) []float64 {
	a := rand.gain * math.Sqrt(6/(rand.n.Min+rand.n.Max))
	rand.n.Min = -a
	rand.n.Max = a
	ret := make([]float64, n)
	for i := 0; i < n; i++ {
		ret[i] = rand.n.Rand()
	}
	return ret
}

func (rand *XavierUniform) RandShape(shapes ...int64) []float64 {
	size := int64(1)
	for _, s := range shapes {
		size *= s
	}
	a := rand.gain * math.Sqrt(6/float64(size))
	rand.n.Min = -a
	rand.n.Max = a
	ret := make([]float64, size)
	for i := int64(0); i < size; i++ {
		ret[i] = rand.n.Rand()
	}
	return ret
}
