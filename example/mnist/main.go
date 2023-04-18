package main

import (
	"fmt"
	"image"
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
		train(trainData, testData)
		return
	}
	model := nextTrain(trainData)
	predict(model, testData)
}

func train(train, test dataSet) {
	initializer := initializer.NewNormal(1, 0.5)

	pt := train.images[0].Bounds().Max

	conv1 := layer.NewConv2D(
		layer.Shape{M: pt.Y, N: pt.X},                   // input shape
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
		input, output := getBatch(train)
		m.Train(input, output)
		if i%100 == 0 {
			loss := m.Loss(input, output)
			acc := accuracy(m, test)
			fmt.Printf("Epoch: %d, Loss: %.05f, Accuracy: %.02f%%\n", i, loss, acc)
			lossPoints = append(lossPoints, plotter.XY{X: float64(i), Y: loss})
			accPoints = append(accPoints, plotter.XY{X: float64(i), Y: acc})
		}
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

func imageData(img image.Image) []float64 {
	pt := img.Bounds().Max
	rows, cols := pt.Y, pt.X
	ret := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			r, _, _, _ := img.At(i, j).RGBA()
			ret[i*cols+j] = float64(r) / 65535
		}
	}
	return ret
}

func onehot(label uint8) []float64 {
	ret := make([]float64, 10)
	ret[label] = 1
	return ret
}

func getBatch(data dataSet) (*mat.Dense, *mat.Dense) {
	max := data.images[0].Bounds().Max
	var input, output []float64
	for i := 0; i < batchSize; i++ {
		n := rand.Intn(len(data.images))
		input = append(input, imageData(data.images[n])...)
		output = append(output, onehot(data.labels[n])...)
	}
	return mat.NewDense(batchSize, max.X*max.Y, input),
		mat.NewDense(batchSize, 10, output)
}

func nextTrain(data dataSet) *model.Model {
	var m model.Model
	runtime.Assert(m.Load(modelFile))
	for i := 0; i < 100; i++ {
		input, output := getBatch(data)
		m.Train(input, output)
		if i%10 == 0 {
			fmt.Printf("Epoch: %d, Loss: %.05f, Accuracy: %.02f%%\n", i,
				m.Loss(input, output), accuracy(&m, data))
		}
	}
	return &m
}

func predict(model *model.Model, data dataSet) {
	var correct int
	var total int
	for i := 0; i < len(data.images); i += batchSize {
		var inputData []float64
		var labels []int
		for i := 0; i < batchSize; i++ {
			inputData = append(inputData, imageData(data.images[i])...)
			labels = append(labels, int(data.labels[i]))
		}
		input := mat.NewDense(batchSize, data.rows*data.cols, inputData)
		pred := model.Predict(input)
		for i := 0; i < batchSize; i++ {
			if getLabel(pred.(vector.RowViewer).RowView(i)) == labels[i] {
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

func accuracy(m *model.Model, data dataSet) float64 {
	var correct int
	var total int
	input, output := getBatch(data)
	pred := m.Predict(input)
	for j := 0; j < batchSize; j++ {
		a := getLabel(pred.(vector.RowViewer).RowView(j))
		b := getLabel(output.RowView(j))
		if a == b {
			correct++
		}
	}
	total += batchSize
	return float64(correct) * 100 / float64(total)
}
