package loss

import (
	"github.com/sugarme/gotch/ts"
)

type Loss interface {
	Loss(y, pred *ts.Tensor) *ts.Tensor
}
