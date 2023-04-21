package main

import (
	"flag"
	"fmt"
	"image"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/internal/utils"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/model"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

const batchSize = 100
const lr = 0.001
const epoch = 10

const dataDir = "./data"
const modelDir = "./model"

func main() {
	pred := flag.String("predict", "", "predict image")
	modelFile := flag.String("model", "./model/latest.model", "model file")
	flag.Parse()

	if len(*pred) > 0 {
		predictImage(*pred, *modelFile)
		return
	}

	// go prof.CpuProfile("./cpu.pprof", 3*time.Minute)
	if _, err := os.Stat(dataDir); os.IsNotExist(err) {
		download()
	}
	fmt.Println("loading train data...")
	trainData := loadData(filepath.Join(dataDir, "train"))
	fmt.Println("loading test data...")
	testData := loadData(filepath.Join(dataDir, "test"))
	os.MkdirAll(modelDir, 0755)
	file := getLatestModel()
	if len(file) == 0 {
		train(&trainData, &testData, testData.rows, testData.cols)
		return
	}
	model := nextTrain(&trainData)
	predict(model, &testData)
}

func train(train, test *dataSet, rows, cols int) {
	initializer := initializer.NewXavierUniform(1)

	conv1 := layer.NewConv2D(
		layer.Shape{M: rows, N: cols},                   // input shape
		layer.Kernel{M: 5, N: 5, InChan: 1, OutChan: 6}, // kernel
		layer.Stride{Y: 1, X: 1},                        // stride
		initializer)
	conv1.SetName("conv1")
	// output: (batch, 28*28*6) => (batch, 4704)

	pool1 := layer.NewMaxPool(
		conv1.OutputShape(),                 // input shape
		layer.Kernel{M: 2, N: 2, InChan: 6}, // kernel
		layer.Stride{Y: 2, X: 2})            // stride
	pool1.SetName("pool1")
	// output: (batch, 14*14*6) => (batch, 1176)

	conv2 := layer.NewConv2D(
		pool1.OutputShape(), // input shape
		layer.Kernel{M: 5, N: 5, InChan: 6, OutChan: 16}, // kernel
		layer.Stride{Y: 1, X: 1},                         // stride
		initializer)
	conv2.SetName("conv2")
	// output: (batch, 14*14*16) => (batch, 3136)

	pool2 := layer.NewMaxPool(
		conv2.OutputShape(),                  // input shape
		layer.Kernel{M: 2, N: 2, InChan: 16}, // kernel shape
		layer.Stride{Y: 2, X: 2})             // stride
	pool2.SetName("pool2")
	// output: (batch, 7*7*16) => (batch, 784)

	var sigmoids []layer.Layer
	for i := 0; i < 4; i++ {
		relu := activation.NewReLU()
		relu.SetName(fmt.Sprintf("sigmoid%d", i+1))
		sigmoids = append(sigmoids, relu)
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
		sigmoids[0],
		pool1,
		conv2,
		sigmoids[1],
		pool2,
		dense1,
		sigmoids[2],
		dense2,
		sigmoids[3],
		output,
	)
	// net.Set(
	// 	layer.NewDense(200, initializer),
	// 	activation.NewReLU(),
	// 	layer.NewDense(100, initializer),
	// 	activation.NewReLU(),
	// 	layer.NewDense(70, initializer),
	// 	activation.NewReLU(),
	// 	layer.NewDense(30, initializer),
	// 	activation.NewReLU(),
	// 	output,
	// )
	loss := loss.NewMSE()
	// optimizer := optimizer.NewSGD(lr, 0)
	optimizer := optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)
	m := model.New(&net, loss, optimizer)
	m.SetName("cnn")

	var lossPoints, accPoints plotter.XYs
	begin := time.Now()
	for i := 0; i < epoch; i++ {
		begin := time.Now()
		trainEpoch(m, train)
		cost := time.Since(begin)
		loss := avgLoss(m, test)
		acc := accuracy(m, test)
		fmt.Printf("\rEpoch: %d, Cost: %s, Loss: %.05f, Accuracy: %.02f%%\n",
			i, cost.String(), loss, acc)
		lossPoints = append(lossPoints, plotter.XY{X: float64(i), Y: loss})
		accPoints = append(accPoints, plotter.XY{X: float64(i), Y: acc})
		m.Save(filepath.Join(modelDir, fmt.Sprintf("%d.model", i)))
		m.Save(filepath.Join(modelDir, "latest.model"))
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
}

func trainEpoch(m *model.Model, data *dataSet) {
	data.Shuffle()
	begin := time.Now()
	for i := 0; i < data.Size(); i += batchSize {
		if i+batchSize > data.Size() {
			break
		}
		input, output := data.Batch(i, batchSize)
		m.Train(input, output)
		fmt.Printf("train: %d/%d, cost: %s\r", i, data.Size(),
			time.Since(begin).String())
	}
}

func nextTrain(data *dataSet) *model.Model {
	var m model.Model
	runtime.Assert(m.Load(filepath.Join(modelDir, "latest.model")))
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
			if getLabel(pred.(utils.DenseRowView).RowView(i)) ==
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
		v := cols.AtVec(i)
		if v > score {
			n = i
			score = v
		}
	}
	return n
}

func avgLoss(m *model.Model, data *dataSet) float64 {
	begin := time.Now()
	input, output := data.All()
	cost := time.Since(begin)
	loss := m.Loss(input, output)
	fmt.Printf("loss cost: %s, loss: %.05f\r", cost.String(), loss)
	return loss
}

func accuracy(m *model.Model, data *dataSet) float64 {
	var correct int
	input, output := data.All()
	begin := time.Now()
	pred := m.Predict(input)
	cost := time.Since(begin)
	for j := 0; j < data.Size(); j++ {
		a := getLabel(pred.(utils.DenseRowView).RowView(j))
		b := getLabel(output.RowView(j))
		if a == b {
			correct++
		}
	}
	fmt.Printf("predict cost: %s, accuracy: %.02f%%\r",
		cost.String(), float64(correct)*100/float64(data.Size()))
	return float64(correct) * 100 / float64(data.Size())
}

func getLatestModel() string {
	dir := filepath.Join(modelDir, "latest.model")
	if _, err := os.Stat(dir); err == nil {
		return dir
	}
	return ""
}

func predictImage(dir, modelFile string) {
	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		fmt.Println("model file not found")
		return
	}
	var m model.Model
	runtime.Assert(m.Load(modelFile))
	m.Print()
	f, err := os.Open(dir)
	runtime.Assert(err)
	defer f.Close()
	img, _, err := image.Decode(f)
	runtime.Assert(err)
	max := img.Bounds().Max
	input := mat.NewDense(1, max.X*max.Y, nil)
	input.SetRow(0, imageData(img))
	output := m.Predict(input)
	fmt.Println("==================================")
	fmt.Println("Predict:")
	v := output.(utils.DenseRowView).RowView(0)
	for i := 0; i < v.Len(); i++ {
		fmt.Printf("%d: %.05f\n", i, v.AtVec(i))
	}
	fmt.Printf("Result: %d\n", getLabel(output.(utils.DenseRowView).RowView(0)))
}
