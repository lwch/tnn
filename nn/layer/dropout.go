package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/params"
	"gonum.org/v1/gonum/mat"
)

type Dropout struct {
	*base
	keepProb float64
}

func NewDropout(keepProb float64) *Dropout {
	var layer Dropout
	layer.keepProb = keepProb
	layer.base = new("dropout", nil, initializer.NewBinomial(1, keepProb))
	return &layer
}

func LoadDropout(name string, params map[string]*pb.Dense, args map[string]*pb.Dense) Layer {
	var layer Dropout
	layer.keepProb = args["kp"].GetData()[0]
	layer.base = new("dropout", nil, initializer.NewBinomial(1, layer.keepProb))
	layer.name = name
	layer.base.loadParams(params)
	return &layer
}

func (layer *Dropout) Name() string {
	return "dropout"
}

func (layer *Dropout) Forward(input mat.Matrix, isTraining bool) (context, output mat.Matrix) {
	if !layer.hasInit {
		layer.initParams()
	}
	rows, cols := input.Dims()
	ctx := mat.NewDense(rows, cols, nil)
	if isTraining {
		ctx.Apply(func(i, j int, v float64) float64 {
			return layer.init.Rand() / layer.keepProb
		}, input)
		var ret mat.Dense
		ret.MulElem(input, ctx)
		return ctx, &ret
	}
	return ctx, input
}

func (layer *Dropout) Backward(context, grad mat.Matrix) (valueGrad mat.Matrix, paramsGrad *params.Params) {
	var ret mat.Dense
	ret.MulElem(grad, context)
	return &ret, nil
}

func (layer *Dropout) Args() map[string]mat.Matrix {
	return map[string]mat.Matrix{
		"kp": mat.NewVecDense(1, []float64{layer.keepProb}),
	}
}
