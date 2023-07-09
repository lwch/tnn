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

func (rand *XavierUniform) Rand() float32 {
	a := rand.gain * math.Sqrt(6/(rand.n.Min+rand.n.Max+1e-9))
	rand.n.Min = -a
	rand.n.Max = a
	return float32(rand.n.Rand())
}

func (rand *XavierUniform) RandN(n int) []float32 {
	a := rand.gain * math.Sqrt(6/(rand.n.Min+rand.n.Max+1e-9))
	rand.n.Min = -a
	rand.n.Max = a
	ret := make([]float32, n)
	for i := 0; i < n; i++ {
		ret[i] = float32(rand.n.Rand())
	}
	return ret
}

func (rand *XavierUniform) RandShape(shapes ...int64) []float32 {
	size := int64(1)
	for _, s := range shapes {
		size *= s
	}
	a := rand.gain * math.Sqrt(6/float64(rand.n.Min+rand.n.Max+1e-9))
	rand.n.Min = -a
	rand.n.Max = a
	ret := make([]float32, size)
	for i := int64(0); i < size; i++ {
		ret[i] = float32(rand.n.Rand())
	}
	return ret
}
