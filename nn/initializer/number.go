package initializer

type Number struct {
	n float32
}

func NewNumber(n float32) *Number {
	return &Number{n: n}
}

func (rand *Number) Rand() float32 {
	return rand.n
}

func (rand *Number) RandN(n int) []float32 {
	ret := make([]float32, n)
	for i := 0; i < n; i++ {
		ret[i] = 1
	}
	return ret
}

func (rand *Number) RandShape(m, n int) []float32 {
	ret := make([]float32, m*n)
	for i := 0; i < m*n; i++ {
		ret[i] = 1
	}
	return ret
}
