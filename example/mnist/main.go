package main

import (
	"fmt"
	"image"
	"math/rand"
	"os"
	"path/filepath"
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
)

const batchSize = 100
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

	hidden1 := layer.NewDense(200, initializer)
	hidden1.SetName("hidden1")
	hidden2 := layer.NewDense(100, initializer)
	hidden2.SetName("hidden2")
	hidden3 := layer.NewDense(70, initializer)
	hidden3.SetName("hidden3")
	hidden4 := layer.NewDense(30, initializer)
	hidden4.SetName("hidden4")
	outputLayer := layer.NewDense(10, initializer)
	outputLayer.SetName("output")

	var net net.Net
	net.Set(
		hidden1,
		activation.NewReLU(),
		hidden2,
		activation.NewReLU(),
		hidden3,
		activation.NewReLU(),
		hidden4,
		activation.NewReLU(),
		outputLayer,
	)
	loss := loss.NewMSE()
	// optimizer := optimizer.NewSGD(lr, 0)
	optimizer := optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)
	m := model.New(&net, loss, optimizer)

	var i int
	for {
		input, output := getBatch(train)
		m.Train(input, output)
		if i%100 == 0 {
			loss := m.Loss(input, output)
			input, output = getBatch(test)
			acc := accuracy(m, input, output)
			fmt.Printf("Epoch: %d, Loss: %.05f, Accuracy: %.02f%%\n", i, loss, acc)
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

func getBatch(data dataSet) (*mat.Dense, *mat.Dense) {
	n := rand.Intn(len(data.images) - batchSize)
	var input, output []float64
	for i := 0; i < batchSize; i++ {
		input = append(input, imageData(data.images[n+i])...)
		output = append(output, onehot(data.labels[n+i])...)
	}
	return mat.NewDense(batchSize, data.rows*data.cols, input),
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

func accuracy(m *model.Model, input, output *mat.Dense) float64 {
	pred := m.Predict(input)
	var correct int
	for i := 0; i < batchSize; i++ {
		var pn, dn int
		var score float64
		for j := 0; j < 10; j++ {
			at := pred.At(i, j)
			if at > score {
				pn = j
				score = at
			}
			if output.At(i, j) > 0.5 {
				dn = j
			}
		}
		if pn == dn {
			correct++
		}
	}
	return float64(correct) * 100 / batchSize
}
