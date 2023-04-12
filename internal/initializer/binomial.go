package initializer

import "gonum.org/v1/gonum/stat/distuv"

type Binomial struct {
	n distuv.Binomial
}

func NewBinomial(n, p float64) *Binomial {
	return &Binomial{
		n: distuv.Binomial{
			N: n,
			P: p,
		},
	}
}

func (rand *Binomial) Rand() float64 {
	return rand.n.Rand()
}

func (rand *Binomial) RandN(n int) []float64 {
	ret := make([]float64, n)
	for i := 0; i < n; i++ {
		ret[i] = rand.n.Rand()
	}
	return ret
}
