package optimizer

import "gorgonia.org/gorgonia"

type Optimizer interface {
	Step(params gorgonia.Nodes) error
}
