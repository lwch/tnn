package initializer

type Zero struct {
}

func NewZero() *Zero {
	return &Zero{}
}

func (rand *Zero) Rand() float32 {
	return 0
}

func (rand *Zero) RandN(n int) []float32 {
	return make([]float32, n)
}

func (rand *Zero) RandShape(m, n int) []float32 {
	return make([]float32, m*n)
}
