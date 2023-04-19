package initializer

import (
	"math"

	"gonum.org/v1/gonum/stat/distuv"
)

type XavierUniform struct {
	n    distuv.Uniform
	gain float64
}

func NewXavierUniform(gain float64) *XavierUniform {
	return &XavierUniform{
		n: distuv.Uniform{
			Min: 0,
			Max: 1,
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

func (rand *XavierUniform) RandShape(m, n int) []float64 {
	a := rand.gain * math.Sqrt(6/float64(m+n))
	rand.n.Min = -a
	rand.n.Max = a
	ret := make([]float64, m*n)
	for i := 0; i < m*n; i++ {
		ret[i] = rand.n.Rand()
	}
	return ret
}
