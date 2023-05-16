package main

import (
	"fmt"
	"math"
	"os"
	"time"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/loss"
	lrs "github.com/lwch/tnn/nn/lr"
	"github.com/lwch/tnn/nn/model"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

const lr = 1e-4
const epoch = 40000
const modelFile = "xor.model"

var input = mat.NewDense(4, 2, []float64{
	0, 0,
	0, 1,
	1, 0,
	1, 1,
})

var output = mat.NewDense(4, 1, []float64{
	0,
	1,
	1,
	0,
})

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

	hidden1 := layer.NewDense(16, initializer)
	hidden1.SetName("hidden1")
	hidden2 := layer.NewDense(8, initializer)
	hidden2.SetName("hidden2")
	hidden3 := layer.NewDense(4, initializer)
	hidden3.SetName("hidden3")
	hidden4 := layer.NewDense(2, initializer)
	hidden4.SetName("hidden4")
	outputLayer := layer.NewDense(1, initializer)
	outputLayer.SetName("output")

	var net net.Net
	net.Set(
		hidden1,
		activation.NewSigmoid(),
		hidden2,
		activation.NewSigmoid(),
		hidden3,
		activation.NewSigmoid(),
		hidden4,
		activation.NewSigmoid(),
		outputLayer,
	)
	loss := loss.NewMSE()
	// optimizer := optimizer.NewSGD(lr, 0)
	optimizer := optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)
	m := model.New(&net, loss, optimizer)
	m.SetLrScheduler(lrs.NewStep(optimizer, 100, 0.999))

	p := plot.New()
	p.Title.Text = "xor train model"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "loss"

	var lossPoints, lrPoints plotter.XYs
	begin := time.Now()
	for i := 0; i < epoch; i++ {
		m.Train(input, output)
		m.ScheduleLr()
		if i%100 == 0 {
			acc := accuracy(m, input, output)
			loss := m.Loss(input, output)
			fmt.Printf("Epoch: %d, Lr: %.05f, Loss: %.05f, Accuracy: %.02f%%\n",
				i, optimizer.GetLr(), loss, acc)
			lossPoints = append(lossPoints, plotter.XY{X: float64(i), Y: loss})
			lrPoints = append(lrPoints, plotter.XY{X: float64(i), Y: optimizer.GetLr() * 1e4})
			if acc >= 100 {
				break
			}
		}
	}
	fmt.Printf("train cost: %s, param count: %d\n",
		time.Since(begin).String(), m.ParamCount())
	fmt.Println("predict:")
	pred := m.Predict(input)
	for i := 0; i < 4; i++ {
		fmt.Printf("%d xor %d: %.2f\n",
			int(input.At(i, 0)), int(input.At(i, 1)),
			pred.At(i, 0))
	}

	lossLine, err := plotter.NewLine(lossPoints)
	runtime.Assert(err)
	lossLine.LineStyle.Color = plotutil.DarkColors[0]

	lrLine, err := plotter.NewLine(lrPoints)
	runtime.Assert(err)
	lrLine.LineStyle.Color = plotutil.DarkColors[1]

	p.Legend.Add("loss", lossLine)
	p.Legend.Add("lr", lrLine)
	p.Legend.XOffs = -20
	p.Legend.YOffs = 5 * vg.Inch

	p.Add(lossLine, lrLine)
	p.Save(8*vg.Inch, 8*vg.Inch, "xor.png")

	runtime.Assert(m.Save(modelFile))
}

func nextTrain() *model.Model {
	var m model.Model
	runtime.Assert(m.Load(modelFile))
	for i := 0; i < 1000; i++ {
		m.Train(input, output)
		if i%100 == 0 {
			fmt.Printf("Epoch: %d, Loss: %.05f, Accuracy: %.02f%%\n", i,
				m.Loss(input, output), accuracy(&m, input, output))
		}
	}
	return &m
}

func predict(model *model.Model) {
	pred := model.Predict(input)
	for i := 0; i < 4; i++ {
		fmt.Printf("%d xor %d: %.2f\n",
			int(input.At(i, 0)), int(input.At(i, 1)),
			pred.At(i, 0))
	}
}

func accuracy(m *model.Model, input, output *mat.Dense) float64 {
	pred := m.Predict(input)
	var correct int
	for i := 0; i < 4; i++ {
		if math.Abs(pred.At(i, 0)-output.At(i, 0)) < math.SmallestNonzeroFloat64 {
			correct++
		}
	}
	return float64(correct) * 100 / 4
}
