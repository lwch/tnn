package loss

import "gorgonia.org/gorgonia"

type Loss interface {
	Loss(y, pred *gorgonia.Node) *gorgonia.Node
}
