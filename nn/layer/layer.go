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
	Forward(input mat.Matrix) mat.Matrix
	Backward(grad mat.Matrix) mat.Matrix
	Params() *params.Params
	Context() params.Params
	Print()
}

type Shape struct {
	M, N int
}

var NoneShape = -1

type forwardFunc func(mat.Matrix) mat.Matrix
type backwardFunc func(mat.Matrix) mat.Matrix

type base struct {
	class    string
	name     string
	shapes   map[string]Shape
	params   params.Params
	input    mat.Dense
	context  params.Params
	init     initializer.Initializer
	hasInit  bool
	forward  forwardFunc
	backward backwardFunc
}

func new(class string, shapes map[string]Shape, init initializer.Initializer,
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
		layer.params[name] = mat.NewDense(shape.M, shape.N, layer.init.RandN(shape.M*shape.N))
		layer.context[name] = mat.NewDense(shape.M, shape.N, nil)
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

func (layer *base) Forward(input mat.Matrix) mat.Matrix {
	layer.input.CloneFrom(input)
	return layer.forward(input)
}

func (layer *base) Backward(grad mat.Matrix) mat.Matrix {
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
	layer.params.Range(func(name string, param mat.Matrix) {
		rows, cols := param.Dims()
		layer.context[name] = mat.NewDense(rows, cols, nil)
	})
	layer.hasInit = true
}

func (layer *base) Print() {
	fmt.Println("  - Name:", layer.Name())
}
