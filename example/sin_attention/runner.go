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

func newRunner(m *model, loss loss.Loss) *runner {
	g := gorgonia.NewGraph()
	x := gorgonia.NewTensor(g, gorgonia.Float32, 3,
		gorgonia.WithName("x"),
		gorgonia.WithShape(batchSize, steps, dims),
		gorgonia.WithInit(gorgonia.Zeroes()))
	y := gorgonia.NewMatrix(g, gorgonia.Float32,
		gorgonia.WithName("y"),
		gorgonia.WithShape(batchSize, 1),
		gorgonia.WithInit(gorgonia.Zeroes()))
	output := x
	var params gorgonia.Nodes
	var ps gorgonia.Nodes
	for _, attn := range m.attn {
		output, ps = attn.Forward(output)
		params = append(params, ps...)
	}
	output = m.flatten.Forward(output)
	output = m.sigmoid.Forward(output)
	pred, ps := m.outputLayer.Forward(output)
	params = append(params, ps...)
	ls := loss.Loss(y, pred)
	_, err := gorgonia.Grad(ls, params...)
	runtime.Assert(err)
	vm := gorgonia.NewTapeMachine(g,
		gorgonia.BindDualValues(params...))
	return &runner{
		g:      g,
		vm:     vm,
		pred:   pred,
		loss:   ls,
		x:      x,
		y:      y,
		params: params,
	}
}

func (r *runner) Train(optimizer optimizer.Optimizer, x, y tensor.Tensor) {
	runtime.Assert(gorgonia.Let(r.x, x))
	runtime.Assert(gorgonia.Let(r.y, y))
	err := r.vm.RunAll()
	if err != nil {
		return
	}
	optimizer.Step(r.params)
	r.vm.Reset()
}

func (r *runner) Predict(x tensor.Tensor) gorgonia.Value {
	runtime.Assert(gorgonia.Let(r.x, x))
	r.vm.RunAll()
	return r.pred.Value()
}

func (r *runner) Loss(x tensor.Tensor) float32 {
	runtime.Assert(gorgonia.Let(r.x, x))
	r.vm.RunAll()
	return r.loss.Value().Data().(float32)
}
