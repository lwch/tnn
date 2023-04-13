package main

import (
	"fmt"
	"os"
	"time"
	"tnn/initializer"
	"tnn/nn/layer"
	"tnn/nn/layer/activation"
	"tnn/nn/loss"
	"tnn/nn/model"
	"tnn/nn/net"
	"tnn/nn/optimizer"

	"github.com/lwch/runtime"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

const lr = 0.001
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
	initializer := initializer.NewNormal(1, 0.5)

	hidden1 := layer.NewDense(100, initializer)
	hidden1.SetName("hidden1")
	hidden2 := layer.NewDense(70, initializer)
	hidden2.SetName("hidden2")
	hidden3 := layer.NewDense(30, initializer)
	hidden3.SetName("hidden3")
	hidden4 := layer.NewDense(10, initializer)
	hidden4.SetName("hidden4")
	outputLayer := layer.NewDense(1, initializer)
	outputLayer.SetName("output")

	var net net.Net
	net.Set(
		hidden1,
		activation.NewSigmoid(),
		layer.NewDropout(0.5),
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

	p := plot.New()
	p.Title.Text = "xor train model"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "loss"

	var points plotter.XYs
	begin := time.Now()
	var i int
	for {
		m.Train(input, output)
		if i%100 == 0 {
			loss := m.Loss(input, output)
			fmt.Printf("Epoch: %d, Loss: %.05f\n", i, loss)
			points = append(points, plotter.XY{X: float64(i), Y: loss})
			if loss < 1e-10 {
				break
			}
		}
		i++
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

	l, err := plotter.NewLine(points)
	runtime.Assert(err)
	l.LineStyle.Color = plotutil.DarkColors[0]
	p.Add(l)
	p.Save(8*vg.Inch, 8*vg.Inch, "xor.png")

	runtime.Assert(m.Save(modelFile))
}

func nextTrain() *model.Model {
	var m model.Model
	runtime.Assert(m.Load(modelFile))
	for i := 0; i < 1000; i++ {
		m.Train(input, output)
		if i%100 == 0 {
			fmt.Printf("Epoch: %d, Loss: %.05f\n", i, m.Loss(input, output))
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
