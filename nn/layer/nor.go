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
	mean := input.MeanAxis(1)
	std := input.VarianceAxis(1)
	rows, _ := input.Dims()
	eps := tensor.New(nil, rows, 1)
	for i := 0; i < rows; i++ {
		eps.Set(i, 0, 1e-9)
	}
	return input.Sub(mean).DivElem(std.Add(eps).Sqrt())
}
