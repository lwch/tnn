package layer

import (
	"tnn/internal/initializer"
	"tnn/internal/nn/optimizer"
	"tnn/internal/shape"

	"gonum.org/v1/gonum/mat"
)

type Linear struct {
	w, b   *mat.Dense
	input  *mat.Dense
	deltaW *mat.Dense
	deltaB *mat.Dense
}

func NewLinear(inputShape shape.Shape, outputN int,
	init initializer.Initializer) *Linear {
	return &Linear{
		w:      mat.NewDense(inputShape.N, outputN, init.RandN(inputShape.N*outputN)),
		b:      mat.NewDense(inputShape.M, outputN, nil),
		input:  mat.NewDense(inputShape.M, inputShape.N, nil),
		deltaW: mat.NewDense(inputShape.N, outputN, nil),
		deltaB: mat.NewDense(inputShape.M, outputN, nil),
	}
}

func (layer *Linear) Name() string {
	return "linear"
}

func (layer *Linear) Forward(input *mat.Dense) *mat.Dense {
	layer.input.Copy(input)
	var ret mat.Dense
	ret.Mul(input, layer.w)
	ret.Add(&ret, layer.b)
	return &ret
}

func (layer *Linear) Backward(grad *mat.Dense) *mat.Dense {
	layer.deltaW.Mul(layer.input.T(), grad)
	layer.deltaB.Add(layer.deltaB, grad)
	var ret mat.Dense
	ret.Mul(grad, layer.w.T())
	return &ret
}

func (layer *Linear) Update(optimizer optimizer.Optimizer) {
	optimizer.Update(layer.w, layer.deltaW)
	optimizer.Update(layer.b, layer.deltaB)
}
