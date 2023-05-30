package layer

import (
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
	mean := input.MeanAxis(0)
	std := input.VarianceAxis(0, false)
	_, cols := input.Dims()
	eps := tensor.New(nil, 1, cols)
	for i := 0; i < cols; i++ {
		eps.Set(0, i, 1e-9)
	}
	return input.Sub(mean).DivElem(std.Add(eps).Sqrt())
}
