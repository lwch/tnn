package main

import (
	"fmt"
	"math"
	"math/rand"
	rt "runtime"
	"sync"

	"github.com/lwch/gotorch/loss"
	"github.com/lwch/gotorch/mmgr"
	"github.com/lwch/gotorch/optimizer"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/runtime"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

const lr = 1e-3
const epoch = 1000
const batchSize = 128
const steps = 32
const dims = 8
const unitSize = steps * dims
const transformerSize = 2

var lossFunc = loss.NewMse
var storage = mmgr.New()

func main() {
	var points []float32
	i := 0.
	for {
		points = append(points, float32(math.Sin(i)))
		i += 0.0001
		if i > 2*math.Pi {
			break
		}
	}

	optm := optimizer.NewAdam(optimizer.WithAdamLr(lr))
	m := newModel(optm)

	p := plot.New()
	p.Title.Text = "predict sin(x)"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "value"

	var miniBatchSize = rt.NumCPU()
	// trainBatchSize = 1

	trainBatch := func(i int) {
		var wg sync.WaitGroup
		wg.Add(miniBatchSize)
		for j := 0; j < miniBatchSize; j++ {
			go func(idx int) {
				defer wg.Done()
				x, y := getBatch(points, idx)
				m.Train(x, y)
			}(i + j)
		}
		wg.Wait()
	}

	var real, predict plotter.XYs
	for i := 0; i < epoch; i += miniBatchSize {
		trainBatch(i)
		m.Apply()
		x, y := getBatch(points, i)
		pred := m.Predict(x)
		ys := y.Float32Value()
		for j := 0; j < len(pred); j += 200 {
			real = append(real, plotter.XY{X: float64(i), Y: float64(ys[j])})
			predict = append(predict, plotter.XY{X: float64(i), Y: float64(pred[j])})
		}
		acc := accuracy(ys, pred)
		fmt.Printf("Epoch: %d, Loss: %e, Accuracy: %.02f%%\n", i, m.Loss(x, y), acc)
		// fmt.Println(y.Value())
		// fmt.Println(pred.Value())
		storage.GC()
	}

	lReal, err := plotter.NewLine(real)
	runtime.Assert(err)
	lReal.LineStyle.Color = plotutil.DarkColors[0]

	lPred, err := plotter.NewLine(predict)
	runtime.Assert(err)
	lPred.LineStyle.Color = plotutil.DarkColors[1]
	lPred.LineStyle.Dashes = []vg.Length{vg.Points(5), vg.Points(1)}

	p.Add(lReal, lPred)
	p.Y.Max = 1.5
	p.Legend.Add("real", lReal)
	p.Legend.Add("predict", lPred)
	p.Legend.Top = true
	p.Legend.XOffs = -20
	p.Save(16*vg.Inch, 4*vg.Inch, "sin.png")
}

func getBatch(points []float32, i int) (*tensor.Tensor, *tensor.Tensor) {
	x := make([]float32, batchSize*unitSize)
	y := make([]float32, batchSize)
	for batch := 0; batch < batchSize; batch++ {
		j := i + batch
		for t := 0; t < unitSize; t++ {
			x[batch*unitSize+t] = points[j%len(points)]
			j++
		}
		y[batch] = points[(i*batchSize+batch)%len(points)]
	}
	rand.Shuffle(batchSize, func(i, j int) {
		dx := make([]float32, unitSize)
		dy := make([]float32, 1)
		copy(dx, x[i*unitSize:(i+1)*unitSize])
		copy(dy, y[i*1:(i+1)*1])
		copy(x[i*unitSize:(i+1)*unitSize], x[j*unitSize:(j+1)*unitSize])
		copy(y[i*1:(i+1)*1], y[j*1:(j+1)*1])
		copy(x[j*unitSize:(j+1)*unitSize], dx)
		copy(y[j*1:(j+1)*1], dy)
	})
	return tensor.FromFloat32(storage, x, batchSize, steps, dims),
		tensor.FromFloat32(storage, y, batchSize, 1)
}

func accuracy(y, pred []float32) float32 {
	var correct float32
	for i := 0; i < batchSize; i++ {
		diff := 1 - float32(math.Abs(float64(pred[i]-y[i])))
		if diff > 0 {
			correct += diff
		}
	}
	return correct * 100 / batchSize
}
