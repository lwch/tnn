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
	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

const lr = 5e-5
const hiddenSize = 10
const epoch = 10000
const modelFile = "xor.model"

var vs = nn.NewVarStore(gotch.CPU)

func main() {
	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		train()
		return
	}
	m := loadModel()
	nextTrain(m)
	predict(m)
}

func train() {
	hidden := layer.NewDense(hiddenSize)
	hidden.SetName("hidden")
	outputLayer := layer.NewDense(1)
	outputLayer.SetName("output")

	net := net.New(vs)
	net.Add(hidden)
	net.Add(activation.NewReLU())
	net.Add(outputLayer)
	loss := loss.NewMSE()
	// optimizer := optimizer.NewSGD(lr, 0)
	optimizer := optimizer.NewAdam(optimizer.WithLearnRate(lr))

	m := newModel(net, loss, optimizer)

	p := plot.New()
	p.Title.Text = "xor train model"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "loss"

	var lossPoints plotter.XYs
	begin := time.Now()
	for i := 0; i < epoch; i++ {
		x, y := getBatch()
		loss := m.Train(x, y)
		if i%10 == 0 {
			acc := accuracy(m)
			fmt.Printf("Epoch: %d, Loss: %e, Accuracy: %.02f%%\n",
				i, loss, acc)
			lossPoints = append(lossPoints, plotter.XY{X: float64(i), Y: float64(loss)})
		}
	}
	fmt.Printf("train cost: %s, param count: %d\n",
		time.Since(begin).String(), net.ParamCount())
	fmt.Println("predict:")
	predict(m)

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

func loadModel() *model {
	net := net.New(vs)
	runtime.Assert(net.Load(modelFile))

	loss := loss.NewMSE()
	optimizer := optimizer.NewAdam(optimizer.WithLearnRate(lr))
	return newModel(net, loss, optimizer)
}

func nextTrain(m *model) {
	for i := 0; i < 1000; i++ {
		x, y := getBatch()
		loss := m.Train(x, y)
		if i%100 == 0 {
			acc := accuracy(m)
			fmt.Printf("Epoch: %d, Loss: %e, Accuracy: %.02f%%\n",
				i, loss, acc)
		}
	}
}

func predict(m *model) {
	x, _ := getBatch()
	xs := x.Vals().([]float32)
	ys := m.Predict(x)
	for i := 0; i < 4; i++ {
		start := i * 2
		fmt.Printf("%d xor %d: %.2f\n",
			int(xs[start]), int(xs[start+1]),
			ys[i])
	}
}

func accuracy(m *model) float32 {
	x, y := getBatch()
	pred := m.Predict(x)
	values := y.Vals().([]float32)
	var correct float32
	for i := 0; i < 4; i++ {
		diff := float32(math.Abs(float64(pred[i]) -
			float64(values[i])))
		if diff < 1 {
			correct += 1 - diff
		}
	}
	return float32(correct) * 100 / 4
}

func getBatch() (*ts.Tensor, *ts.Tensor) {
	x, err := ts.NewTensorFromData([][]float32{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}, []int64{4, 2})
	runtime.Assert(err)
	y, err := ts.NewTensorFromData([][]float32{
		{0},
		{1},
		{1},
		{0},
	}, []int64{4, 1})
	runtime.Assert(err)
	return x, y
}
