package layer

import (
	"fmt"
	"sync"

	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/params"
	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	SetName(string)
	Name() string
	Class() string
	Forward(input mat.Matrix, isTraining bool) (context, output mat.Matrix)
	Backward(context, grad mat.Matrix) (valueGrad mat.Matrix, paramsGrad *params.Params)
	Params() *params.Params
	Args() map[string]mat.Matrix
	Print()
}

type Shape struct {
	M, N int
}

type Kernel struct {
	M, N            int
	InChan, OutChan int
}

type Stride struct {
	X, Y int
}

var NoneShape = -1

type base struct {
	class   string
	name    string
	shapes  map[string]Shape
	params  *params.Params
	init    initializer.Initializer
	mInit   sync.Mutex
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
		layer.params.InitWithData(name, shape.M, shape.N, layer.init.RandShape(shape.M, shape.N))
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

func (layer *base) loadParams(ps map[string]*pb.Dense) {
	layer.params.Load(ps)
	layer.shapes = make(map[string]Shape)
	layer.params.Range(func(name string, dense mat.Matrix) {
		rows, cols := dense.Dims()
		layer.shapes[name] = Shape{M: rows, N: cols}
	})
	layer.hasInit = true
}

func (layer *base) Print() {
	fmt.Println("  - Class:", layer.Class())
	fmt.Println("    Name:", layer.Name())
}

func (layer *base) Args() map[string]mat.Matrix {
	return nil
}
