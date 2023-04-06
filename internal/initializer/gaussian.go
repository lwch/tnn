package initializer

import (
	"time"

	rng "github.com/leesper/go_rng"
)

type Gaussian struct {
	g            *rng.GaussianGenerator
	mean, stddev float64
}

func NewGaussian(mean, stddev float64) *Gaussian {
	return &Gaussian{
		g:      rng.NewGaussianGenerator(time.Now().UnixNano()),
		mean:   mean,
		stddev: stddev,
	}
}

func (g *Gaussian) Get() float64 {
	return g.g.Gaussian(g.mean, g.stddev)
}
