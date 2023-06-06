package main

import (
	"fmt"
	"math"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/optimizer"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

const lr = 1e-4
const epoch = 1000
const batchSize = 2
const steps = 8
const featureSize = 1
const unitSize = steps * featureSize
const hiddenSize = 2
const clearSteps = 1 // 每隔多少次迭代清空一次隐藏层状态，提高反向传播速度

func main() {
	var points []float32
	i := 0.
	for {
		points = append(points, float32(math.Sin(i)))
		i += 0.001
		if i > 2*math.Pi {
			break
		}
	}

	g := gorgonia.NewGraph()

	m := newModel()

	loss := loss.NewMSE()
	optimizer := optimizer.NewAdam(optimizer.WithLearnRate(lr))

	p := plot.New()
	p.Title.Text = "predict sin(x)"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "value"

	x := gorgonia.NewTensor(g, tensor.Float32, 3,
		gorgonia.WithShape(batchSize, steps, featureSize), gorgonia.WithName("x"))
	y := gorgonia.NewTensor(g, tensor.Float32, 2,
		gorgonia.WithShape(batchSize, 1), gorgonia.WithName("y"))

	var real, predict plotter.XYs
	for i := 0; i < epoch; i++ {
		input, output := getBatch(points, i+batchSize)
		runtime.Assert(gorgonia.Let(x, input))
		runtime.Assert(gorgonia.Let(y, output))
		m.Train(i, loss, optimizer, x, y)

		pred := m.Predict(x)
		y1 := y.Value().Data().([]float32)[0]
		y2 := pred.Data().([]float32)[0]
		real = append(real, plotter.XY{X: float64(i), Y: float64(y1)})
		predict = append(predict, plotter.XY{X: float64(i), Y: float64(y2)})
		if i%10 == 0 {
			acc := accuracy(m, x, y)
			loss := m.Loss(loss, x, y)
			fmt.Printf("Epoch: %d, Loss: %e, Accuracy: %.02f%%\n", i, loss, acc)
			// fmt.Println(y.Value())
			// fmt.Println(pred.Value())
		}
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

// func getBatch(i int) (tensor.Tensor, tensor.Tensor) {
// 	inputs := make([]float32, batchSize*steps*featureSize)
// 	outputs := make([]float32, batchSize)
// 	max := float64(epoch * batchSize * steps * featureSize)
// 	sampleSize := steps * featureSize
// 	for batch := 0; batch < batchSize; batch++ {
// 		var n float64
// 		for t := 0; t < steps; t++ {
// 			for f := 0; f < featureSize; f++ {
// 				n = float64(i) / max * 100
// 				inputs[batch*sampleSize+t*featureSize+f] = float32(math.Sin(n))
// 				i++
// 			}
// 		}
// 		n = float64(i) / max * 100
// 		outputs[batch] = float32(math.Sin(n))
// 	}
// 	return tensor.New(tensor.WithShape(batchSize, steps, featureSize), tensor.WithBacking(inputs)),
// 		tensor.New(tensor.WithShape(batchSize, 1), tensor.WithBacking(outputs))
// }

func getBatch(points []float32, i int) (tensor.Tensor, tensor.Tensor) {
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
	// rand.Shuffle(batchSize, func(i, j int) {
	// 	dx := make([]float32, unitSize)
	// 	dy := make([]float32, 1)
	// 	copy(dx, x[i*unitSize:(i+1)*unitSize])
	// 	copy(dy, y[i*1:(i+1)*1])
	// 	copy(x[i*unitSize:(i+1)*unitSize], x[j*unitSize:(j+1)*unitSize])
	// 	copy(y[i*1:(i+1)*1], y[j*1:(j+1)*1])
	// 	copy(x[j*unitSize:(j+1)*unitSize], dx)
	// 	copy(y[j*1:(j+1)*1], dy)
	// })
	return tensor.New(tensor.WithShape(batchSize, steps, featureSize), tensor.WithBacking(x)),
		tensor.New(tensor.WithShape(batchSize, 1), tensor.WithBacking(y))
}

func accuracy(m *model, x, y *gorgonia.Node) float32 {
	pred := m.Predict(x)
	y1Values := y.Value().Data().([]float32)
	y2Values := pred.Data().([]float32)
	var correct float32
	for i := 0; i < batchSize; i++ {
		diff := 1 - float32(math.Abs(float64(y1Values[i])-
			float64(y2Values[i])))
		if diff > 0 {
			correct += diff
		}
	}
	return correct * 100 / batchSize
}
