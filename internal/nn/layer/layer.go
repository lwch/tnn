package layer

import (
	"tnn/internal/initializer"

	"gonum.org/v1/gonum/mat"
)

type Params map[string]*mat.Dense

func (params *Params) Copy(ps Params) {
	*params = make(Params)
	for name, value := range ps {
		var dense mat.Dense
		dense.CloneFrom(value)
		(*params)[name] = &dense
	}
}

func (params Params) Add(grads *Params) {
	for name, grad := range *grads {
		p := params[name]
		if p == nil {
			continue
		}
		p.Add(p, grad)
	}
}

type Layer interface {
	Name() string
	Forward(input *mat.Dense) *mat.Dense
	Backward(grad *mat.Dense) *mat.Dense
	Params() *Params
	Context() Params
}

type shape struct {
	m, n int
}

var noneShape = -1

type base struct {
	shapes  map[string]shape
	params  Params
	input   mat.Dense
	context Params
	init    initializer.Initializer
	hasInit bool
}

func new(shapes map[string]shape, init initializer.Initializer) *base {
	return &base{
		shapes:  shapes,
		context: make(Params),
		init:    init,
	}
}

func (layer *base) initParams() {
	if layer.hasInit {
		return
	}
	layer.params = make(Params)
	for name := range layer.shapes {
		shape := layer.shapes[name]
		layer.params[name] = mat.NewDense(shape.m, shape.n, layer.init.RandN(shape.m*shape.n))
		layer.context[name] = mat.NewDense(shape.m, shape.n, nil)
	}
	layer.hasInit = true
}

func (layer *base) Params() *Params {
	return &layer.params
}

func (layer *base) Context() Params {
	return layer.context
}
