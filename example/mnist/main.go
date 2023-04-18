package main

import (
	"fmt"
	"image"
	"math/rand"
	"os"
	"path/filepath"

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
)

const batchSize = 100
const lr = 0.01

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
		layer.Shape{M: pt.Y, N: pt.X}, // input shape
		layer.Shape{M: 10, N: 10},     // kernel shape
		layer.Stride{Y: 1, X: 1},      // stride
		initializer)
	conv1.SetName("conv1")

	pool1 := layer.NewMaxPool(
		conv1.OutputShape(),      // input shape
		layer.Shape{M: 2, N: 2},  // kernel shape
		layer.Stride{Y: 2, X: 2}) // stride
	pool1.SetName("pool1")

	conv2 := layer.NewConv2D(
		pool1.OutputShape(),      // input shape
		layer.Shape{M: 5, N: 5},  // kernel shape
		layer.Stride{Y: 1, X: 1}, // stride
		initializer)
	conv2.SetName("conv2")

	pool2 := layer.NewMaxPool(
		conv2.OutputShape(),      // input shape
		layer.Shape{M: 2, N: 2},  // kernel shape
		layer.Stride{Y: 2, X: 2}) // stride

	var relus []layer.Layer
	for i := 0; i < 4; i++ {
		relu := activation.NewReLU()
		relu.SetName(fmt.Sprintf("relu%d", i+1))
		relus = append(relus, relu)
	}

	var net net.Net
	net.Set(
		conv1,
		relus[0],
		pool1,
		conv2,
		relus[1],
		pool2,
		layer.NewDense(120, initializer),
		relus[2],
		layer.NewDense(84, initializer),
		relus[3],
		layer.NewDense(10, initializer),
	)
	loss := loss.NewSoftmax(1)
	// optimizer := optimizer.NewSGD(lr, 0)
	optimizer := optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)
	m := model.New(&net, loss, optimizer)

	var i int
	for {
		input, output := getBatch(train)
		m.Train(input, output)
		loss := m.Loss(input, output)
		acc := accuracy(m, test)
		fmt.Printf("Epoch: %d, Loss: %.05f, Accuracy: %.02f%%\n", i, loss, acc)
		// points = append(points, plotter.XY{X: float64(i), Y: loss})
		// if acc >= 100 {
		// 	break
		// }
		i++
	}
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
	// for i := 0; i < 1000; i++ {
	// 	m.Train(input, output)
	// 	if i%100 == 0 {
	// 		fmt.Printf("Epoch: %d, Loss: %.05f, Accuracy: %.02f%%\n", i,
	// 			m.Loss(input, output), m.Accuracy(input, output))
	// 	}
	// }
	return &m
}

func predict(model *model.Model, data dataSet) {

}

func accuracy(m *model.Model, data dataSet) float64 {
	get := func(cols mat.Vector) int {
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

	var correct int
	var total int
	input, output := getBatch(data)
	pred := m.Predict(input)
	for j := 0; j < batchSize; j++ {
		a := get(pred.(vector.RowViewer).RowView(j))
		b := get(output.RowView(j))
		if a == b {
			correct++
		}
	}
	total += batchSize
	return float64(correct) * 100 / float64(total)
}
