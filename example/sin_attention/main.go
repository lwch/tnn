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
	"github.com/lwch/tnn/nn/optimizer"
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

const lr = 1e-3
const epoch = 10000
const batchSize = 10
const times = 8
const transformerSize = 32

func main() {
	initializer := initializer.NewXavierUniform(1)

	var layers []layer.Layer
	for i := 0; i < transformerSize; i++ {
		layers = append(layers, addTransformer(initializer)...)
	}
	layers = append(layers, activation.NewReLU())
	layers = append(layers, layer.NewDense(1, initializer))

	loss := loss.NewMSE()
	optimizer := optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)

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
				input, output := getBatch(i * batchSize * times)
				pred := forward(layers, input, true)
				grad := loss.Loss(pred, output)
				grad.ZeroGrad()
				grad.Backward(grad)
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
		input, output := getBatch(i * batchSize * times)
		<-trained
		params := getParams(layers)
		optimizer.Update(params)
		pred := forward(layers, input, false)
		real = append(real, plotter.XY{X: float64(i), Y: output.Value().At(0, 0)})
		predict = append(predict, plotter.XY{X: float64(i), Y: pred.Value().At(0, 0)})
		if i%10 == 0 {
			acc := accuracy(layers, input, output)
			loss := loss.Loss(input, output)
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

func addTransformer(init initializer.Initializer) []layer.Layer {
	var layers []layer.Layer
	layers = append(layers, layer.NewSelfAttention(times, init))
	layers = append(layers, layer.NewNor())
	layers = append(layers, layer.NewDense(times*4, init))
	layers = append(layers, activation.NewReLU())
	layers = append(layers, layer.NewDense(times, init))
	layers = append(layers, layer.NewNor())
	return layers
}

func forwardTransformer(layers []layer.Layer, i int, x *tensor.Tensor, train bool) (*tensor.Tensor, int) {
	y := layers[i].Forward(x, train)
	y = layers[i+1].Forward(y, train) // nor
	y = layers[i+2].Forward(y, train) // dense
	y = layers[i+3].Forward(y, train) // relu
	y = layers[i+4].Forward(y, train) // dense
	y = layers[i+5].Forward(y, train) // nor
	return y, i + 6
}

func forward(layers []layer.Layer, x *tensor.Tensor, train bool) *tensor.Tensor {
	i := 0
	var y *tensor.Tensor
	for j := 0; j < transformerSize; j++ {
		y, i = forwardTransformer(layers, i, x, train)
	}
	y = layers[i].Forward(y, train)   // relu
	y = layers[i+1].Forward(y, train) // output
	return y
}

func getParams(layers []layer.Layer) []*params.Params {
	var ret []*params.Params
	for _, layer := range layers {
		params := layer.Params()
		if params.IsEmpty() {
			continue
		}
		ret = append(ret, params)
	}
	return ret
}

func getBatch(i int) (*tensor.Tensor, *tensor.Tensor) {
	inputs := tensor.New(nil, batchSize, times)
	outputs := tensor.New(nil, batchSize, 1)
	max := float64(epoch * batchSize * times)
	for batch := 0; batch < batchSize; batch++ {
		var n float64
		for t := 0; t < times; t++ {
			n = float64(i) / max * 100
			inputs.Set(batch, t, math.Sin(n))
			i++
		}
		n = float64(i) / max * 100
		outputs.Set(batch, 0, math.Sin(n))
	}
	return inputs, outputs
}

func accuracy(layers []layer.Layer, input, output *tensor.Tensor) float64 {
	predict := forward(layers, input, false)
	var correct float64
	for i := 0; i < batchSize; i++ {
		diff := 1 - math.Abs(output.Value().At(i, 0)-predict.Value().At(i, 0))
		if diff > 0 {
			correct += diff
		}
	}
	return correct * 100 / batchSize
}
