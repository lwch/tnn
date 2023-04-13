package layer

import (
	"fmt"
	"tnn/initializer"
	"tnn/nn/params"
	"tnn/nn/pb"

	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	SetName(string)
	Name() string
	Class() string
	Forward(input *mat.Dense) *mat.Dense
	Backward(grad *mat.Dense) *mat.Dense
	Params() *params.Params
	Context() params.Params
	Print()
}

type shape struct {
	m, n int
}

var noneShape = -1

type forwardFunc func(*mat.Dense) *mat.Dense
type backwardFunc func(*mat.Dense) *mat.Dense

type base struct {
	class    string
	name     string
	shapes   map[string]shape
	params   params.Params
	input    mat.Dense
	context  params.Params
	init     initializer.Initializer
	hasInit  bool
	forward  forwardFunc
	backward backwardFunc
}

func new(class string, shapes map[string]shape, init initializer.Initializer,
	forward forwardFunc, backward backwardFunc) *base {
	return &base{
		class:    class,
		shapes:   shapes,
		params:   make(params.Params),
		context:  make(params.Params),
		init:     init,
		forward:  forward,
		backward: backward,
	}
}

func (layer *base) initParams() {
	if layer.hasInit {
		return
	}
	for name := range layer.shapes {
		shape := layer.shapes[name]
		layer.params[name] = mat.NewDense(shape.m, shape.n, layer.init.RandN(shape.m*shape.n))
		layer.context[name] = mat.NewDense(shape.m, shape.n, nil)
	}
	layer.hasInit = true
}

func (layer *base) Class() string {
	return layer.class
}

func (layer *base) SetName(name string) {
	layer.name = name
}

func (layer *base) Name() string {
	if len(layer.name) == 0 {
		return layer.class
	}
	return layer.name
}

func (layer *base) Forward(input *mat.Dense) *mat.Dense {
	layer.input.CloneFrom(input)
	return layer.forward(input)
}

func (layer *base) Backward(grad *mat.Dense) *mat.Dense {
	return layer.backward(grad)
}

func (layer *base) Params() *params.Params {
	return &layer.params
}

func (layer *base) Context() params.Params {
	return layer.context
}

func (layer *base) loadParams(ps map[string]*pb.Dense) {
	layer.params.Load(ps)
	layer.context = make(params.Params)
	layer.params.Range(func(name string, param *mat.Dense) {
		rows, cols := param.Dims()
		layer.context[name] = mat.NewDense(rows, cols, nil)
	})
	layer.hasInit = true
}

func (layer *base) Print() {
	fmt.Println("  - Name:", layer.Name())
}
