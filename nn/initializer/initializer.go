package initializer

type Initializer interface {
	Rand() float32
	RandN(n int) []float32
	RandShape(shapes ...int64) []float32
}
