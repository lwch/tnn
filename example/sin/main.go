package main

import (
	"fmt"
	"math"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/model"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

const lr = 1e-3
const epoch = 3000
const batchSize = 10
const steps = 8
const featureSize = 16
const hiddenSize = 32

func main() {
	g := gorgonia.NewGraph()
	net := net.New(
		layer.NewRnn(g, featureSize, steps, hiddenSize),
		// layer.NewLstm(g, 1, times, 32),
		layer.NewFlatten(g),
		layer.NewDense(g, hiddenSize*steps, 1),
	)
	loss := loss.NewMSE()
	optimizer := optimizer.NewAdam(lr, 0, 0)
	m := model.New(net, loss, optimizer)

	p := plot.New()
	p.Title.Text = "predict sin(x)"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "value"

	x := gorgonia.NewTensor(g, tensor.Float32, 3,
		gorgonia.WithShape(batchSize, steps, featureSize), gorgonia.WithName("x"))
	y := gorgonia.NewTensor(g, tensor.Float32, 2,
		gorgonia.WithShape(batchSize, 1), gorgonia.WithName("y"))

	pred := m.Compile(g, x, y)

	var real, predict plotter.XYs
	for i := 0; i < epoch; i++ {
		input, output := getBatch(i * batchSize * steps * featureSize)
		runtime.Assert(gorgonia.Let(x, input))
		runtime.Assert(gorgonia.Let(y, output))
		runtime.Assert(m.Train())
		y1 := y.Value().Data().([]float32)[0]
		y2 := pred.Value().Data().([]float32)[0]
		real = append(real, plotter.XY{X: float64(i), Y: float64(y1)})
		predict = append(predict, plotter.XY{X: float64(i), Y: float64(y2)})
		if i%10 == 0 {
			acc := accuracy(y.Value(), pred.Value())
			loss := m.Loss()
			fmt.Printf("Epoch: %d, Loss: %e, Accuracy: %.02f%%\n", i, loss, acc)
			// fmt.Println(y.Value())
			// fmt.Println(pred.Value())
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

func getBatch(i int) (tensor.Tensor, tensor.Tensor) {
	inputs := make([]float32, batchSize*steps*featureSize)
	outputs := make([]float32, batchSize)
	max := float64(epoch * batchSize * steps * featureSize)
	sampleSize := steps * featureSize
	for batch := 0; batch < batchSize; batch++ {
		var n float64
		for t := 0; t < steps; t++ {
			for f := 0; f < featureSize; f++ {
				n = float64(i) / max * 100
				inputs[batch*sampleSize+t*featureSize+f] = float32(math.Sin(n))
				i++
			}
		}
		n = float64(i) / max * 100
		outputs[batch] = float32(math.Sin(n))
	}
	return tensor.New(tensor.WithShape(batchSize, steps, featureSize), tensor.WithBacking(inputs)),
		tensor.New(tensor.WithShape(batchSize, 1), tensor.WithBacking(outputs))
}

func accuracy(y1, y2 gorgonia.Value) float32 {
	y1Values := y1.Data().([]float32)
	y2Values := y2.Data().([]float32)
	var correct float32
	for i := 0; i < batchSize; i++ {
		diff := 1 - float32(math.Abs(float64(y1Values[i])-
			float64(y2Values[i])))
		if diff > 0 {
			correct += diff
		}
	}
	return correct * 100 / batchSize
}
