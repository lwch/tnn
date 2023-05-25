package layer

import (
	"github.com/lwch/tnn/internal/math"
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/tensor"
)

type Nor struct {
	*base
}

func NewNor() Layer {
	var layer Nor
	layer.base = new("nor", nil, nil)
	return &layer
}

func LoadNor(name string, params map[string]*pb.Dense, _ map[string]*pb.Dense) Layer {
	var layer Nor
	layer.base = new("nor", nil, nil)
	layer.name = name
	return &layer
}

func (layer *Nor) Forward(input *tensor.Tensor, isTraining bool) *tensor.Tensor {
	min := math.Min(input)
	rows, cols := input.Dims()
	return input.Sub(tensor.Numbers(rows, cols, min)).Inv()
}
