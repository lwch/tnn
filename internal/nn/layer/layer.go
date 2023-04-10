package layer

import (
	"fmt"
	"tnn/internal/initializer"

	"gonum.org/v1/gonum/mat"
)

type Params map[string]*mat.Dense

func (params *Params) Add(grad *mat.Dense) {
	for name := range *params {
		p := (*params)[name]
		fmt.Println(name)
		fmt.Println(p.Dims())
		fmt.Println(grad.Dims())
		p.Add(p, grad)
		(*params)[name] = p
	}
}

type Layer interface {
	Name() string
	Forward(input *mat.Dense) *mat.Dense
	Backward(grad *mat.Dense) *mat.Dense
	Params() *Params
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

func (layer *base) initParams(init initializer.Initializer) {
	if layer.hasInit {
		return
	}
	layer.params = make(Params)
	for name := range layer.shapes {
		shape := layer.shapes[name]
		layer.params[name] = mat.NewDense(shape.m, shape.n, init.RandN(shape.m*shape.n))
		layer.context[name] = mat.NewDense(shape.m, shape.n, nil)
	}
	layer.hasInit = true
}

func (layer *base) Params() *Params {
	return &layer.params
}
