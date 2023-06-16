package schedule

import "math"

type Schedule struct {
	dims   int
	warmup int
	steps  int
	lr     float64
}

func New(dims, warmup int) *Schedule {
	s := &Schedule{
		dims:   dims,
		warmup: warmup,
		steps:  1,
	}
	s.Step()
	return s
}

func (s *Schedule) Step() {
	a := math.Pow(float64(s.steps), -0.5)
	b := float64(s.steps) * math.Pow(float64(s.warmup), -1.5)
	s.steps++
	s.lr = math.Pow(float64(s.dims), -0.5) * math.Min(a, b)
}

func (s *Schedule) Get() float64 {
	return s.lr
}
