package initializer

import (
	"math"

	"gonum.org/v1/gonum/stat/distuv"
)

type XavierUniform struct {
	n    distuv.Uniform
	gain float32
}

func NewXavierUniform(gain float32) *XavierUniform {
	return &XavierUniform{
		n: distuv.Uniform{
			Min: 0,
			Max: 1,
		},
		gain: gain,
	}
}

func (rand *XavierUniform) Rand() float32 {
	a := rand.gain * float32(math.Sqrt(6/(rand.n.Min+rand.n.Max)))
	rand.n.Min = float64(-a)
	rand.n.Max = float64(a)
	return float32(rand.n.Rand())
}

func (rand *XavierUniform) RandN(n int) []float32 {
	a := rand.gain * float32(math.Sqrt(6/(rand.n.Min+rand.n.Max)))
	rand.n.Min = float64(-a)
	rand.n.Max = float64(a)
	ret := make([]float32, n)
	for i := 0; i < n; i++ {
		ret[i] = float32(rand.n.Rand())
	}
	return ret
}

func (rand *XavierUniform) RandShape(m, n int) []float32 {
	a := rand.gain * float32(math.Sqrt(6/float64(m+n)))
	rand.n.Min = float64(-a)
	rand.n.Max = float64(a)
	ret := make([]float32, m*n)
	for i := 0; i < m*n; i++ {
		ret[i] = float32(rand.n.Rand())
	}
	return ret
}
