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
	mean := math.Mean(input)
	std := math.Var(input)
	rows, cols := input.Dims()
	means := tensor.New(nil, rows, cols)
	stds := tensor.New(nil, rows, cols)
	eps := tensor.New(nil, rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			means.Set(i, j, mean[i])
			stds.Set(i, j, std[i])
			eps.Set(i, j, 1e-6)
		}
	}
	return input.Sub(means).DivElem(stds.Add(eps).Sqrt())
}
