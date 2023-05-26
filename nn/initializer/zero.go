package initializer

type Zero struct {
}

func NewZero() *Zero {
	return &Zero{}
}

func (rand *Zero) Rand() float64 {
	return 0
}

func (rand *Zero) RandN(n int) []float64 {
	return make([]float64, n)
}

func (rand *Zero) RandShape(m, n int) []float64 {
	return make([]float64, m*n)
}
