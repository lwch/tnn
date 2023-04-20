package initializer

type Initializer interface {
	Rand() float64
	RandN(n int) []float64
	RandShape(m, n int) []float64
}
