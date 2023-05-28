package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
)

type Dropout struct {
	*base
	rate float64
	init *initializer.Binomial
}

func NewDropout(rate float64) Layer {
	var layer Dropout
	layer.base = new("dropout", nil, nil)
	layer.rate = rate
	layer.init = initializer.NewBinomial(1, rate)
	return &layer
}

func LoadDropout(name string, params map[string]*pb.Dense, args map[string]*pb.Dense) Layer {
	arr := args["rate"].GetData()
	var layer Dropout
	layer.base = new("dropout", nil, nil)
	layer.rate = arr[0]
	layer.init = initializer.NewBinomial(1, layer.rate)
	layer.name = name
	layer.base.loadParams(params)
	return &layer
}

func (layer *Dropout) Forward(input *tensor.Tensor, isTraining bool) *tensor.Tensor {
	if isTraining {
		rows, cols := input.Dims()
		arr := layer.init.RandShape(rows, cols)
		for i := range arr {
			arr[i] /= layer.rate
		}
		return input.MulElem(tensor.New(arr, rows, cols))
	}
	return input
}

func (layer *Dropout) Args() map[string]*mat.VecDense {
	return map[string]*mat.VecDense{
		"rate": mat.NewVecDense(1, []float64{float64(layer.rate)}),
	}
}
