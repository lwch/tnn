package sample

type Sample struct {
	x []int
	y int
}

func New(x []int, y int) *Sample {
	return &Sample{x: x, y: y}
}
