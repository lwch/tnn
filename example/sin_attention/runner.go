package main

import (
	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/optimizer"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type runner struct {
	g      *gorgonia.ExprGraph
	vm     gorgonia.VM
	pred   *gorgonia.Node
	loss   *gorgonia.Node
	x      *gorgonia.Node
	y      *gorgonia.Node
	params gorgonia.Nodes
}

func newRunner() *runner {
	g := gorgonia.NewGraph()
	return &runner{
		g: g,
		x: gorgonia.NewTensor(g, gorgonia.Float32, 3,
			gorgonia.WithName("x"),
			gorgonia.WithShape(batchSize, steps, dims),
			gorgonia.WithInit(gorgonia.Zeroes())),
		y: gorgonia.NewMatrix(g, gorgonia.Float32,
			gorgonia.WithName("y"),
			gorgonia.WithShape(batchSize, 1),
			gorgonia.WithInit(gorgonia.Zeroes())),
	}
}

func (r *runner) Compile(m *model, loss loss.Loss) {
	output := r.x
	var ps gorgonia.Nodes
	for _, attn := range m.attn {
		output, ps = attn.Forward(output)
		r.params = append(r.params, ps...)
	}
	output = m.flatten.Forward(output)
	output = m.sigmoid.Forward(output)
	r.pred, ps = m.outputLayer.Forward(output)
	r.params = append(r.params, ps...)
	r.loss = loss.Loss(r.y, r.pred)
	_, err := gorgonia.Grad(r.loss, r.params...)
	runtime.Assert(err)
	prog, locMap, err := gorgonia.Compile(r.g)
	runtime.Assert(err)
	r.vm = gorgonia.NewTapeMachine(r.g,
		gorgonia.WithPrecompiled(prog, locMap),
		gorgonia.BindDualValues(r.params...))
}

func (r *runner) Train(optimizer optimizer.Optimizer, x, y tensor.Tensor) {
	runtime.Assert(gorgonia.Let(r.x, x))
	runtime.Assert(gorgonia.Let(r.y, y))
	r.vm.Reset()
	runtime.Assert(r.vm.RunAll())
	runtime.Assert(optimizer.Step(r.params))
}

func (r *runner) Predict(x tensor.Tensor) gorgonia.Value {
	runtime.Assert(gorgonia.Let(r.x, x))
	r.vm.Reset()
	runtime.Assert(r.vm.RunAll())
	return r.pred.Value()
}

func (r *runner) Loss(x tensor.Tensor) float32 {
	runtime.Assert(gorgonia.Let(r.x, x))
	r.vm.Reset()
	runtime.Assert(r.vm.RunAll())
	return r.loss.Value().Data().(float32)
}
