package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/internal/utils"
	"github.com/lwch/tnn/nn/initializer"
	"gonum.org/v1/gonum/mat"
)

type Dropout struct {
	*base
	keepProb float64
}

func NewDropout(keepProb float64) *Dropout {
	var layer Dropout
	layer.keepProb = keepProb
	layer.base = new("dropout", map[string]Shape{
		"m": {NoneShape, NoneShape},
	}, initializer.NewBinomial(1, keepProb), layer.forward, layer.backward)
	return &layer
}

func LoadDropout(name string, params map[string]*pb.Dense, args map[string]*pb.Dense) Layer {
	var layer Dropout
	layer.keepProb = args["kp"].GetData()[0]
	layer.base = new("dropout", nil, initializer.NewBinomial(1, layer.keepProb),
		layer.forward, layer.backward)
	layer.name = name
	layer.base.loadParams(params)
	return &layer
}

func (layer *Dropout) Name() string {
	return "dropout"
}

func (layer *Dropout) forward(input mat.Matrix) mat.Matrix {
	if !layer.hasInit {
		rows, cols := input.Dims()
		layer.shapes["m"] = Shape{rows, cols}
		layer.initParams()
	}
	if layer.isTraining {
		m := layer.context.Get("m")
		m.(utils.DenseApply).Apply(func(i, j int, v float64) float64 {
			return layer.init.Rand() / layer.keepProb
		}, m)
		var ret mat.Dense
		ret.MulElem(input, m)
		return &ret
	}
	return input
}

func (layer *Dropout) backward(grad mat.Matrix) mat.Matrix {
	dm := layer.context.Get("m")

	var ret mat.Dense
	ret.MulElem(grad, dm)
	return &ret
}

func (layer *Dropout) Args() map[string]mat.Matrix {
	return map[string]mat.Matrix{
		"kp": mat.NewVecDense(1, []float64{layer.keepProb}),
	}
}
