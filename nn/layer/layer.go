package layer

import (
	"gorgonia.org/gorgonia"
)

type Layer interface {
	Forward(x *gorgonia.Node) *gorgonia.Node
	Params() gorgonia.Nodes
}
