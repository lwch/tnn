package main

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/initializer"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/model"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"github.com/lwch/tnn/nn/vector"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

const batchSize = 100
const lr = 0.01
const epoch = 10000

const dataDir = "./data"
const modelFile = "mnist.model"

func main() {
	// go prof.CpuProfile("./cpu.pprof", 3*time.Minute)
	if _, err := os.Stat(dataDir); os.IsNotExist(err) {
		download()
	}
	fmt.Println("loading train data...")
	trainData := loadData(filepath.Join(dataDir, "train"))
	fmt.Println("loading test data...")
	testData := loadData(filepath.Join(dataDir, "test"))
	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		train(&trainData, &testData, testData.rows, testData.cols)
		return
	}
	model := nextTrain(&trainData)
	predict(model, &testData)
}

func train(train, test *dataSet, rows, cols int) {
	initializer := initializer.NewNormal(1, 0.5)

	conv1 := layer.NewConv2D(
		layer.Shape{M: rows, N: cols},                   // input shape
		layer.Kernel{M: 5, N: 5, InChan: 1, OutChan: 6}, // kernel
		layer.Stride{Y: 1, X: 1},                        // stride
		initializer)
	conv1.SetName("conv1")
	// output: (100, 28*28*6) => (100, 4704)

	pool1 := layer.NewMaxPool(
		conv1.OutputShape(),                             // input shape
		layer.Kernel{M: 2, N: 2, InChan: 6, OutChan: 6}, // kernel
		layer.Stride{Y: 2, X: 2})                        // stride
	pool1.SetName("pool1")
	// output: (100, 14*14*6) => (100, 1176)

	conv2 := layer.NewConv2D(
		pool1.OutputShape(), // input shape
		layer.Kernel{M: 5, N: 5, InChan: 6, OutChan: 16}, // kernel
		layer.Stride{Y: 1, X: 1},                         // stride
		initializer)
	conv2.SetName("conv2")
	// output: (100, 14*14*16) => (100, 3136)

	pool2 := layer.NewMaxPool(
		conv2.OutputShape(), // input shape
		layer.Kernel{M: 2, N: 2, InChan: 16, OutChan: 16}, // kernel shape
		layer.Stride{Y: 2, X: 2})                          // stride
	pool2.SetName("pool2")
	// output: (100, 7*7*16) => (100, 784)

	var relus []layer.Layer
	for i := 0; i < 4; i++ {
		relu := activation.NewReLU()
		relu.SetName(fmt.Sprintf("relu%d", i+1))
		relus = append(relus, relu)
	}

	dense1 := layer.NewDense(120, initializer)
	dense1.SetName("dense1")

	dense2 := layer.NewDense(84, initializer)
	dense2.SetName("dense2")

	output := layer.NewDense(10, initializer)
	output.SetName("output")

	var net net.Net
	net.Set(
		conv1,
		relus[0],
		pool1,
		conv2,
		relus[1],
		pool2,
		dense1,
		relus[2],
		dense2,
		relus[3],
		output,
	)
	loss := loss.NewSoftmax(1)
	// optimizer := optimizer.NewSGD(lr, 0)
	optimizer := optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)
	m := model.New(&net, loss, optimizer)

	var lossPoints, accPoints plotter.XYs
	begin := time.Now()
	for i := 0; i < epoch; i++ {
		begin := time.Now()
		trainEpoch(m, train)
		cost := time.Since(begin)
		loss := avgLoss(m, test)
		acc := accuracy(m, test)
		fmt.Printf("Epoch: %d, Cost: %s, Loss: %.05f, Accuracy: %.02f%%\n",
			i, cost.String(), loss, acc)
		lossPoints = append(lossPoints, plotter.XY{X: float64(i), Y: loss})
		accPoints = append(accPoints, plotter.XY{X: float64(i), Y: acc})
	}
	fmt.Printf("train cost: %s, param count: %d\n",
		time.Since(begin).String(), m.ParamCount())

	p := plot.New()
	p.Title.Text = "mnist cnn model"
	p.X.Label.Text = "epoch"

	nor := func(args plotter.XYs) plotter.XYs {
		var max float64
		for _, point := range args {
			if point.Y > max {
				max = point.Y
			}
		}
		var ret plotter.XYs
		for _, point := range args {
			point.Y = point.Y * 100 / max
			ret = append(ret, point)
		}
		return ret
	}
	l1, err := plotter.NewLine(nor(lossPoints))
	runtime.Assert(err)
	l1.LineStyle.Color = plotutil.DarkColors[0]

	l2, err := plotter.NewLine(accPoints)
	runtime.Assert(err)
	l2.LineStyle.Color = plotutil.DarkColors[1]

	p.Add(l1, l2)
	p.Legend.Add("loss", l1)
	p.Legend.Add("accurcy", l2)
	p.Legend.XOffs = -20
	p.Legend.YOffs = 6 * vg.Inch
	p.Save(8*vg.Inch, 8*vg.Inch, "mnist.png")

	runtime.Assert(m.Save(modelFile))
}

func trainEpoch(m *model.Model, data *dataSet) {
	data.Shuffle()
	for i := 0; i < data.Size(); i += batchSize {
		if i+batchSize > data.Size() {
			break
		}
		input, output := data.Batch(i, batchSize)
		m.Train(input, output)
	}
}

func nextTrain(data *dataSet) *model.Model {
	var m model.Model
	runtime.Assert(m.Load(modelFile))
	for i := 0; i < 100; i++ {
		input, output := data.Batch(rand.Intn(data.Size()), batchSize)
		begin := time.Now()
		m.Train(input, output)
		cost := time.Since(begin)
		if i%10 == 0 {
			fmt.Printf("Epoch: %d, Cost: %s, Loss: %.05f, Accuracy: %.02f%%\n", i,
				cost.String(), m.Loss(input, output), accuracy(&m, data))
		}
	}
	return &m
}

func predict(model *model.Model, data *dataSet) {
	var correct int
	var total int
	for i := 0; i < len(data.images); i += batchSize {
		if i+batchSize > data.Size() {
			break
		}
		input, output := data.Batch(i, batchSize)
		pred := model.Predict(input)
		for i := 0; i < batchSize; i++ {
			if getLabel(pred.(vector.RowViewer).RowView(i)) ==
				getLabel(output.RowView(i)) {
				correct++
			}
		}
		total += batchSize
	}
	fmt.Printf("Predict Accuracy: %.02f%%\n",
		float64(correct)*100/float64(total))
}

func getLabel(cols mat.Vector) int {
	var n int
	var score float64
	for i := 0; i < cols.Len(); i++ {
		v := cols.At(i, 0)
		if v > score {
			n = i
			score = v
		}
	}
	return n
}

func avgLoss(m *model.Model, data *dataSet) float64 {
	var sum float64
	var cnt float64
	for i := 0; i < data.Size(); i += batchSize {
		if i+batchSize > data.Size() {
			break
		}
		input, output := data.Batch(i, batchSize)
		sum += m.Loss(input, output)
		cnt++
	}
	return sum / cnt
}

func accuracy(m *model.Model, data *dataSet) float64 {
	var correct int
	var total int
	for i := 0; i < data.Size(); i += batchSize {
		if i+batchSize > data.Size() {
			break
		}
		input, output := data.Batch(i, batchSize)
		pred := m.Predict(input)
		for j := 0; j < batchSize; j++ {
			a := getLabel(pred.(vector.RowViewer).RowView(j))
			b := getLabel(output.RowView(j))
			if a == b {
				correct++
			}
		}
		total += batchSize
	}
	return float64(correct) * 100 / float64(total)
}
