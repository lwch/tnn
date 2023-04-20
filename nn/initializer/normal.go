package initializer

import "gonum.org/v1/gonum/stat/distuv"

type Normal struct {
	n distuv.Normal
}

func NewNormal(mean, stddev float64) *Normal {
	return &Normal{
		n: distuv.Normal{
			Mu:    mean,
			Sigma: stddev,
		},
	}
}

func (rand *Normal) Rand() float64 {
	return rand.n.Rand()
}

func (rand *Normal) RandN(n int) []float64 {
	ret := make([]float64, n)
	for i := 0; i < n; i++ {
		ret[i] = rand.n.Rand()
	}
	return ret
}

func (rand *Normal) RandShape(m, n int) []float64 {
	ret := make([]float64, m*n)
	for i := 0; i < m*n; i++ {
		ret[i] = rand.n.Rand()
	}
	return ret
}
