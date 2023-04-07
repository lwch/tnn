package layer

import (
	"tnn/internal/initializer"
	"tnn/internal/nn/optimizer"

	"gonum.org/v1/gonum/mat"
)

type Linear struct {
	input  mat.Dense
	w, b   *mat.Dense
	deltaW mat.Dense
	deltaB *mat.Dense
}

func NewLinear(inputM, inputN, outputN int, init initializer.Initializer) *Linear {
	return &Linear{
		w:      mat.NewDense(inputN, outputN, init.RandN(inputN*outputN)),
		b:      mat.NewDense(inputM, outputN, nil),
		deltaB: mat.NewDense(inputM, outputN, nil),
	}
}

func (layer *Linear) Name() string {
	return "linear"
}

func (layer *Linear) Forward(input *mat.Dense) *mat.Dense {
	layer.input.CloneFrom(input)
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
	optimizer.Update(layer.w, &layer.deltaW)
	optimizer.Update(layer.b, layer.deltaB)
}
