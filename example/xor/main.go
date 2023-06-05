package main

import (
	"fmt"
	"math"
	"os"
	"time"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/loss"
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
const hiddenSize = 10
const epoch = 40000
const modelFile = "xor.model"

var m = newModel()

var xData = tensor.New(tensor.WithShape(4, 2), tensor.WithBacking([]float32{
	0, 0,
	0, 1,
	1, 0,
	1, 1,
}))

var yData = tensor.New(tensor.WithShape(4, 1), tensor.WithBacking([]float32{
	0,
	1,
	1,
	0,
}))

func main() {
	defer m.Close()
	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		train()
		return
	}
	x, y := loadModel()
	nextTrain(x, y)
	predict()
}

func train() {
	var x = gorgonia.NewMatrix(m.G(), tensor.Float32,
		gorgonia.WithShape(xData.Shape()...),
		gorgonia.WithName("x"))
	var y = gorgonia.NewMatrix(m.G(), tensor.Float32,
		gorgonia.WithShape(yData.Shape()...),
		gorgonia.WithName("y"))

	hidden := layer.NewDense(m.G(), 2, hiddenSize)
	hidden.SetName("hidden")
	outputLayer := layer.NewDense(m.G(), hiddenSize, yData.Shape()[1])
	outputLayer.SetName("output")

	net := net.New(m.G())
	net.Add(hidden)
	net.Add(activation.NewReLU())
	net.Add(outputLayer)
	loss := loss.NewMSE()
	// optimizer := optimizer.NewSGD(lr, 0)
	optimizer := optimizer.NewAdam(lr, 0, 0)

	m.Compile(net, loss, x, y)

	p := plot.New()
	p.Title.Text = "xor train model"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "loss"

	var lossPoints plotter.XYs
	begin := time.Now()
	for i := 0; i < epoch; i++ {
		gorgonia.Let(x, xData)
		gorgonia.Let(y, yData)
		loss := m.Train(optimizer)
		if i%100 == 0 {
			acc := accuracy()
			fmt.Printf("Epoch: %d, Loss: %e, Accuracy: %.02f%%\n",
				i, loss, acc)
			lossPoints = append(lossPoints, plotter.XY{X: float64(i), Y: float64(loss.Data().(float32))})
			if acc >= 100 {
				break
			}
		}
	}
	fmt.Printf("train cost: %s, param count: %d\n",
		time.Since(begin).String(), net.ParamCount())
	fmt.Println("predict:")
	predict()

	lossLine, err := plotter.NewLine(lossPoints)
	runtime.Assert(err)
	lossLine.LineStyle.Color = plotutil.DarkColors[0]

	p.Legend.Add("loss", lossLine)
	p.Legend.XOffs = -20
	p.Legend.YOffs = 5 * vg.Inch

	p.Add(lossLine)
	p.Save(8*vg.Inch, 8*vg.Inch, "xor.png")

	runtime.Assert(net.Save(modelFile))
}

func loadModel() (*gorgonia.Node, *gorgonia.Node) {
	net := net.New(m.G())
	runtime.Assert(net.Load(modelFile))

	loss := loss.NewMSE()

	var x = gorgonia.NewMatrix(m.G(), tensor.Float32,
		gorgonia.WithShape(xData.Shape()...),
		gorgonia.WithName("x"))
	var y = gorgonia.NewMatrix(m.G(), tensor.Float32,
		gorgonia.WithShape(yData.Shape()...),
		gorgonia.WithName("y"))

	m.Compile(net, loss, x, y)
	return x, y
}

func nextTrain(x, y *gorgonia.Node) {
	optimizer := optimizer.NewAdam(lr, 0, 0)

	for i := 0; i < 1000; i++ {
		gorgonia.Let(x, xData)
		gorgonia.Let(y, yData)
		loss := m.Train(optimizer)
		if i%100 == 0 {
			acc := accuracy()
			fmt.Printf("Epoch: %d, Loss: %e, Accuracy: %.02f%%\n",
				i, loss, acc)
		}
	}
}

func predict() {
	pred, _ := m.Evaluate()
	xs := xData.Data().([]float32)
	ys := pred.Data().([]float32)
	for i := 0; i < 4; i++ {
		start := i * 2
		fmt.Printf("%d xor %d: %.2f\n",
			int(xs[start]), int(xs[start+1]),
			ys[i])
	}
}

func accuracy() float32 {
	pred, _ := m.Evaluate()
	predValues := pred.Data().([]float32)
	values := yData.Data().([]float32)
	var correct float32
	for i := 0; i < 4; i++ {
		diff := float32(math.Abs(float64(predValues[i]) -
			float64(values[i])))
		if diff < 1 {
			correct += 1 - diff
		}
	}
	return float32(correct) * 100 / 4
}
