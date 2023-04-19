package initializer

import "gonum.org/v1/gonum/stat/distuv"

type Uniform struct {
	n distuv.Uniform
}

func NewUniform(min, max float64) *Uniform {
	return &Uniform{
		n: distuv.Uniform{
			Min: min,
			Max: max,
		},
	}
}

func (rand *Uniform) Rand() float64 {
	return rand.n.Rand()
}

func (rand *Uniform) RandN(n int) []float64 {
	ret := make([]float64, n)
	for i := 0; i < n; i++ {
		ret[i] = rand.n.Rand()
	}
	return ret
}

func (rand *Uniform) RandShape(m, n int) []float64 {
	ret := make([]float64, m*n)
	for i := 0; i < m*n; i++ {
		ret[i] = rand.n.Rand()
	}
	return ret
}
