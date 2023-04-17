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
	"gonum.org/v1/gonum/mat"
)

const batchSize = 1
const lr = 0.1

const dataDir = "./data"
const modelFile = "mnist.model"

func init() {
	rand.Seed(time.Now().UnixNano())
}

func main() {
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
	initializer := initializer.NewNormal(1, 0)

	pt := train.images[0].Bounds().Max

	conv1 := layer.NewConv2D(
		layer.Shape{M: pt.Y, N: pt.X}, // input shape
		layer.Shape{M: 5, N: 5},       // kernel shape
		layer.Stride{Y: 1, X: 1},      // stride
		initializer)
	conv1.SetName("conv1")

	var net net.Net
	net.Set(
		conv1,
		activation.NewReLU(),
		layer.NewDense(10, initializer),
	)
	loss := loss.NewMSE()
	optimizer := optimizer.NewSGD(lr, 0)
	// optimizer := optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)
	m := model.New(&net, loss, optimizer)

	var i int
	for {
		input, output := getBatch(train, i%(len(train.images)-batchSize))
		m.Train(input, output)
		if i%100 == 0 {
			// loss := m.Loss(input, output)
			// acc := accuracy(m, test)
			// fmt.Printf("Epoch: %d, Loss: %.05f, Accuracy: %.02f%%\n", i, loss, acc)
			// points = append(points, plotter.XY{X: float64(i), Y: loss})
			// if acc >= 100 {
			// 	break
			// }
		}
		i++
	}
}

func imageData(img image.Image) []float64 {
	pt := img.Bounds().Max
	rows, cols := pt.X, pt.Y
	ret := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			r, _, _, _ := img.At(i, j).RGBA()
			ret[i*rows+j] = float64(r) / 65535
		}
	}
	return ret
}

func onehot(label uint8) []float64 {
	ret := make([]float64, 10)
	ret[label] = 1
	return ret
}

func getBatch(data dataSet, n int) (*mat.VecDense, *mat.Dense) {
	var input, output []float64
	for i := 0; i < batchSize; i++ {
		input = append(input, imageData(data.images[n+i])...)
		output = append(output, onehot(data.labels[n+i])...)
	}
	return mat.NewVecDense(len(input), input),
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
	_ = get

	var correct int
	var total int
	for i := 0; i < len(data.images); i += batchSize {
		if i+batchSize > len(data.images) {
			break
		}
		input, output := getBatch(data, i)
		// pred := m.Predict(input)
		// for j := 0; j < batchSize; j++ {
		// 	if get(pred.RowView(j)) == get(output.RowView(j)) {
		// 		correct++
		// 	}
		// }
		_ = input
		_ = output
		total += batchSize
	}
	return float64(correct) * 100 / float64(total)
}
