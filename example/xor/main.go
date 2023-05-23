package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

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

const lr = 1e-3
const epoch = 40000
const modelFile = "xor.model"

var input = [][]float64{
	{0, 0},
	{0, 1},
	{1, 0},
	{1, 1},
}

var output = []float64{
	0,
	1,
	1,
	0,
}

func main() {
	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		train()
		return
	}
	model := nextTrain()
	predict(model)
}

func train() {
	initializer := initializer.NewXavierUniform(1)

	hidden1 := layer.NewDense(10, initializer)
	hidden1.SetName("hidden1")
	outputLayer := layer.NewDense(1, initializer)
	outputLayer.SetName("output")

	var net net.Net
	net.Set(
		hidden1,
		activation.NewReLU(),
		outputLayer,
	)
	loss := loss.NewMSE()
	// optimizer := optimizer.NewSGD(lr, 0)
	optimizer := optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)
	m := model.New(&net, loss, optimizer)

	p := plot.New()
	p.Title.Text = "xor train model"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "loss"

	var lossPoints plotter.XYs
	begin := time.Now()
	for i := 0; i < epoch; i++ {
		inputs, outputs := getBatch()
		m.Train(inputs, outputs)
		if i%100 == 0 {
			acc := accuracy(m, inputs, outputs)
			loss := m.Loss(inputs, outputs)
			fmt.Printf("Epoch: %d, Lr: %.05f, Loss: %e, Accuracy: %.02f%%\n",
				i, optimizer.GetLr(), loss, acc)
			lossPoints = append(lossPoints, plotter.XY{X: float64(i), Y: loss})
			if acc >= 100 {
				break
			}
		}
	}
	fmt.Printf("train cost: %s, param count: %d\n",
		time.Since(begin).String(), m.ParamCount())
	fmt.Println("predict:")
	inputs, _ := getBatch()
	pred := m.Predict(inputs)
	for i := 0; i < 4; i++ {
		fmt.Printf("%d xor %d: %.2f\n",
			int(inputs.Value().At(i, 0)), int(inputs.Value().At(i, 1)),
			pred.Value().At(i, 0))
	}

	lossLine, err := plotter.NewLine(lossPoints)
	runtime.Assert(err)
	lossLine.LineStyle.Color = plotutil.DarkColors[0]

	p.Legend.Add("loss", lossLine)
	p.Legend.XOffs = -20
	p.Legend.YOffs = 5 * vg.Inch

	p.Add(lossLine)
	p.Save(8*vg.Inch, 8*vg.Inch, "xor.png")

	runtime.Assert(m.Save(modelFile))
}

func getBatch() (*tensor.Tensor, *tensor.Tensor) {
	idx := make([]int, len(input))
	for i := range idx {
		idx[i] = i
	}
	rand.Shuffle(len(idx), func(i, j int) {
		idx[i], idx[j] = idx[j], idx[i]
	})
	inputs := make([]float64, len(input)*2)
	outputs := make([]float64, len(output))
	for i := 0; i < len(input); i++ {
		inputs[i*2] = input[idx[i]][0]
		inputs[i*2+1] = input[idx[i]][1]
		outputs[i] = output[idx[i]]
	}
	input := tensor.New(inputs, len(input), 2)
	input.SetName("input")
	output := tensor.New(outputs, len(output), 1)
	output.SetName("output")
	return input, output
}

func nextTrain() *model.Model {
	var m model.Model
	runtime.Assert(m.Load(modelFile))
	for i := 0; i < 1000; i++ {
		inputs, outputs := getBatch()
		m.Train(inputs, outputs)
		if i%100 == 0 {
			fmt.Printf("Epoch: %d, Loss: %e, Accuracy: %.02f%%\n", i,
				m.Loss(inputs, outputs), accuracy(&m, inputs, outputs))
		}
	}
	return &m
}

func predict(model *model.Model) {
	inputs, _ := getBatch()
	pred := model.Predict(inputs)
	for i := 0; i < 4; i++ {
		fmt.Printf("%d xor %d: %.2f\n",
			int(inputs.Value().At(i, 0)), int(inputs.Value().At(i, 1)),
			pred.Value().At(i, 0))
	}
}

func accuracy(m *model.Model, input, output *tensor.Tensor) float64 {
	pred := m.Predict(input)
	var correct float64
	for i := 0; i < 4; i++ {
		diff := math.Abs(pred.Value().At(i, 0) - output.Value().At(i, 0))
		if diff < 1 {
			correct += 1 - diff
		}
	}
	return float64(correct) * 100 / 4
}
