package layer

import (
	"sync"

	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
)

type Layer interface {
	Params() *params.Params
	Forward(input *tensor.Tensor, isTraining bool) *tensor.Tensor
}

type Shape struct {
	M, N int
}

var NoneShape = -1

type Kernel struct {
	M, N            int
	InChan, OutChan int
}

type Stride struct {
	X, Y int
}

type base struct {
	class   string
	name    string
	shapes  map[string]Shape
	mInit   sync.Mutex
	params  *params.Params
	init    initializer.Initializer
	hasInit bool
}

func new(class string, shapes map[string]Shape, init initializer.Initializer) *base {
	return &base{
		class:  class,
		shapes: shapes,
		params: params.New(),
		init:   init,
	}
}

func (layer *base) initParams() {
	layer.mInit.Lock()
	defer layer.mInit.Unlock()
	if layer.hasInit {
		return
	}
	for name := range layer.shapes {
		shape := layer.shapes[name]
		t := tensor.New(layer.init.RandShape(shape.M, shape.N), shape.M, shape.N)
		t.SetName(layer.Name() + "." + name)
		layer.params.Set(name, t)
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

func (layer *base) Params() *params.Params {
	return layer.params
}
