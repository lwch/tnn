package main

import (
	"flag"
	"fmt"
	"image"
	"image/png"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	rt "runtime"
	"sync"
	"sync/atomic"
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
	// go prof.CpuProfile("./cpu.pprof", 3*time.Minute)
	pred := flag.String("predict", "", "predict image")
	modelFile := flag.String("model", "./model/latest.model", "model file")
	exportConv := flag.Bool("export-conv", false, "export conv layers image")
	flag.Parse()

	if len(*pred) > 0 {
		predictImage(*pred, *modelFile)
		return
	}
	if *exportConv {
		exportConvImages(*modelFile)
		return
	}

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
	model := nextTrain(&trainData, &testData)
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
		relu := activation.NewSigmoid()
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
		conv1,       // output: (batch, 28*28*6) => (batch, 4704)
		sigmoids[0], // output: (batch, 4704)
		pool1,       // output: (batch, 14*14*6) => (batch, 1176)
		conv2,       // output: (batch, 14*14*16) => (batch, 3136)
		sigmoids[1], // output: (batch, 3136)
		pool2,       // output: (batch, 7*7*16) => (batch, 784)
		dense1,      // output: (batch, 120)
		sigmoids[2], // output: (batch, 120)
		dense2,      // output: (batch, 84)
		sigmoids[3], // output: (batch, 84)
		output,      // output: (batch, 10)
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
	loss := loss.NewSoftmax(1)
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

type context struct {
	m       *model.Model
	data    *dataSet
	begin   time.Time
	arrived atomic.Uint64
	total   uint64
	input   chan int
}

func worker(ctx *context) {
	for {
		i, ok := <-ctx.input
		if !ok {
			return
		}
		input, output := ctx.data.Batch(i, batchSize)
		ctx.m.Train(input, output)
		ctx.arrived.Add(uint64(batchSize))
		fmt.Printf("train: %d/%d, cost: %s\r", ctx.arrived.Load(), ctx.total,
			time.Since(ctx.begin).String())
	}
}

func trainEpoch(m *model.Model, data *dataSet) {
	data.Shuffle()
	var ctx context
	ctx.m = m
	ctx.data = data
	ctx.begin = time.Now()
	ctx.total = uint64(data.Size())
	ctx.input = make(chan int, data.Size()/batchSize+1)

	var wg sync.WaitGroup
	wg.Add(rt.NumCPU())
	for i := 0; i < rt.NumCPU(); i++ {
		go func() {
			defer wg.Done()
			worker(&ctx)
		}()
	}

	for i := 0; i < data.Size(); i += batchSize {
		if i+batchSize > data.Size() {
			break
		}
		ctx.input <- i
	}
	close(ctx.input)
	wg.Wait()
}

func nextTrain(trainData, testData *dataSet) *model.Model {
	var m model.Model
	runtime.Assert(m.Load(filepath.Join(modelDir, "latest.model")))
	for i := 0; i < 100; i++ {
		input, output := trainData.Batch(rand.Intn(trainData.Size()), batchSize)
		begin := time.Now()
		m.Train(input, output)
		cost := time.Since(begin)
		if i%10 == 0 {
			fmt.Printf("Epoch: %d, Cost: %s, Loss: %.05f, Accuracy: %.02f%%\n", i,
				cost.String(), m.Loss(input, output), accuracy(&m, testData))
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
	var sum float64
	var cnt float64
	begin := time.Now()
	for i := 0; i < data.Size(); i += batchSize {
		if i+batchSize > data.Size() {
			break
		}
		input, output := data.Batch(i, batchSize)
		sum += m.Loss(input, output)
		cnt++
		fmt.Printf("loss: %d/%d, cost: %s, loss: %.05f\r", i, data.Size(),
			time.Since(begin).String(), sum/cnt)
	}
	return sum / cnt
}

func accuracy(m *model.Model, data *dataSet) float64 {
	var correct int
	var total int
	begin := time.Now()
	for i := 0; i < data.Size(); i += batchSize {
		if i+batchSize > data.Size() {
			break
		}
		input, output := data.Batch(i, batchSize)
		pred := m.Predict(input)
		for j := 0; j < batchSize; j++ {
			a := getLabel(pred.(utils.DenseRowView).RowView(j))
			b := getLabel(output.RowView(j))
			if a == b {
				correct++
			}
		}
		total += batchSize
		fmt.Printf("predict: %d/%d, cost: %s, accuracy: %.02f%%\r", i, data.Size(),
			time.Since(begin).String(), float64(correct)*100/float64(total))
	}
	return float64(correct) * 100 / float64(total)
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

func exportConvImages(modelFile string) {
	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		fmt.Println("model file not found")
		return
	}
	var m model.Model
	runtime.Assert(m.Load(modelFile))
	os.RemoveAll("./conv")
	runtime.Assert(os.MkdirAll("./conv", 0755))

	nor := func(data []float64) []uint8 {
		min := math.MaxFloat64
		for i := 0; i < len(data); i++ {
			if data[i] < min {
				min = data[i]
			}
		}
		ret := make([]uint8, len(data))
		for i := 0; i < len(data); i++ {
			ret[i] = uint8((data[i] - min) * 255)
		}
		return ret
	}
	save := func(img image.Image, name string) {
		f, err := os.Create(filepath.Join("./conv", fmt.Sprintf("%s.png", name)))
		runtime.Assert(err)
		defer f.Close()
		runtime.Assert(png.Encode(f, img))
	}

	for _, l := range m.Layers() {
		if l.Class() != "conv2d" {
			continue
		}
		layer := l.(*layer.Conv2D)
		ps := layer.Params()
		var tmp mat.Dense
		tmp.CloneFrom(ps.Get("w"))
		pb := ps.Get("b")
		db := pb.(utils.DenseRowView).RowView(0)
		rows, _ := tmp.Dims()
		for i := 0; i < rows; i++ {
			row := tmp.RowView(i)
			row.(utils.AddVec).AddVec(row, db)
		}
		rows, cols := tmp.Dims()
		img := image.NewGray(image.Rect(0, 0, cols, rows))
		copy(img.Pix, nor(tmp.RawMatrix().Data))
		save(img, layer.Name())
	}
}
