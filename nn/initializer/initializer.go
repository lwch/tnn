package initializer

type Initializer interface {
	Rand() float32
	RandN(n int) []float32
	RandShape(m, n int) []float32
}
