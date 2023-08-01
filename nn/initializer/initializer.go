package initializer

import "github.com/lwch/gotorch/tensor"

type Initializer interface {
	Init(*tensor.Tensor)
}
