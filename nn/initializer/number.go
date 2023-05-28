package initializer

type Number struct {
	n float64
}

func NewNumber(n float64) *Number {
	return &Number{n: n}
}

func (rand *Number) Rand() float64 {
	return rand.n
}

func (rand *Number) RandN(n int) []float64 {
	ret := make([]float64, n)
	for i := 0; i < n; i++ {
		ret[i] = 1
	}
	return ret
}

func (rand *Number) RandShape(m, n int) []float64 {
	ret := make([]float64, m*n)
	for i := 0; i < m*n; i++ {
		ret[i] = 1
	}
	return ret
}
