package main

import (
	"fmt"
	"math"
	rt "runtime"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/model"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

const lr = 0.001
const epoch = 300
const batchSize = 10
const seqSize = 8
const dims = 8
const heads = 4
const unitSize = seqSize * dims
const transformerSize = 1

func main() {
	init := initializer.NewXavierUniform(1)

	var net net.Net
	for i := 0; i < transformerSize; i++ {
		// self attention
		net.Add(layer.NewSelfAttention(seqSize, dims, heads, init))
		net.Add(layer.NewNor())
		// FNN
		net.Add(layer.NewDense(unitSize, init))
		net.Add(activation.NewReLU())
		net.Add(layer.NewDense(unitSize, init))
		net.Add(layer.NewNor())
	}
	// 回归
	net.Add(activation.NewReLU())
	net.Add(layer.NewDense(1, init))

	loss := loss.NewSoftmax()
	optimizer := optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)

	m := model.New(&net, loss, optimizer)

	p := plot.New()
	p.Title.Text = "predict sin(x)"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "value"

	ch := make(chan int)
	trained := make(chan struct{})

	for i := 0; i < rt.NumCPU(); i++ {
		go func() {
			for {
				i := <-ch
				input, output := getBatch(i * batchSize * unitSize)
				pred := m.Forward(input, true)
				m.Backward(pred, output)
				trained <- struct{}{}
			}
		}()
	}

	go func() {
		for i := 0; i < epoch; i++ {
			ch <- i
		}
	}()

	var real, predict plotter.XYs
	for i := 0; i < epoch; i++ {
		input, output := getBatch(i * batchSize * unitSize)
		<-trained
		pred := m.Forward(input, false)
		real = append(real, plotter.XY{X: float64(i), Y: output.Value().At(0, 0)})
		predict = append(predict, plotter.XY{X: float64(i), Y: pred.Value().At(0, 0)})
		if i%10 == 0 {
			m.Apply()
			pred.ZeroGrad()
			pred = m.Forward(input, false)
			acc := accuracy(m, input, output)
			loss := loss.Loss(pred, output)
			fmt.Printf("Epoch: %d, Loss: %e, Accuracy: %.02f%%\n",
				i, loss.Value().At(0, 0), acc)
			// fmt.Println(mat.Formatted(output.Value()))
			// fmt.Println(mat.Formatted(pred.Value()))
		}
	}

	lReal, err := plotter.NewLine(real)
	runtime.Assert(err)
	lReal.LineStyle.Color = plotutil.DarkColors[0]

	lPred, err := plotter.NewLine(predict)
	runtime.Assert(err)
	lPred.LineStyle.Color = plotutil.DarkColors[1]
	lPred.LineStyle.Dashes = []vg.Length{vg.Points(10), vg.Points(5)}

	p.Add(lReal, lPred)
	p.Y.Max = 1.5
	p.Legend.Add("real", lReal)
	p.Legend.Add("predict", lPred)
	p.Legend.Top = true
	p.Legend.XOffs = -20
	p.Save(16*vg.Inch, 4*vg.Inch, "sin.png")
}

func getBatch(i int) (*tensor.Tensor, *tensor.Tensor) {
	inputs := tensor.New(nil, batchSize, unitSize)
	outputs := tensor.New(nil, batchSize, 1)
	max := float64(epoch * batchSize * unitSize)
	for batch := 0; batch < batchSize; batch++ {
		var n float64
		for t := 0; t < unitSize; t++ {
			n = float64(i) / max * 100
			inputs.Set(batch, t, math.Sin(n))
			i++
		}
		n = float64(i) / max * 100
		outputs.Set(batch, 0, math.Sin(n))
	}
	return inputs, outputs
}

func accuracy(m *model.Model, input, output *tensor.Tensor) float64 {
	predict := m.Forward(input, false)
	var correct float64
	for i := 0; i < batchSize; i++ {
		diff := 1 - math.Abs(output.Value().At(i, 0)-predict.Value().At(i, 0))
		if diff > 0 {
			correct += diff
		}
	}
	return correct * 100 / batchSize
}
