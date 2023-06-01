package initializer

import "gonum.org/v1/gonum/stat/distuv"

type Binomial struct {
	n distuv.Binomial
}

func NewBinomial(n, p float32) *Binomial {
	return &Binomial{
		n: distuv.Binomial{
			N: float64(n),
			P: float64(p),
		},
	}
}

func (rand *Binomial) Rand() float32 {
	return float32(rand.n.Rand())
}

func (rand *Binomial) RandN(n int) []float32 {
	ret := make([]float32, n)
	for i := 0; i < n; i++ {
		ret[i] = float32(rand.n.Rand())
	}
	return ret
}

func (rand *Binomial) RandShape(m, n int) []float32 {
	ret := make([]float32, m*n)
	for i := 0; i < m*n; i++ {
		ret[i] = float32(rand.n.Rand())
	}
	return ret
}
