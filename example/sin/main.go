package main

// import (
// 	"fmt"
// 	"math"

// 	"github.com/lwch/runtime"
// 	"github.com/lwch/tnn/nn/loss"
// 	"github.com/lwch/tnn/nn/optimizer"
// 	"github.com/sugarme/gotch"
// 	"github.com/sugarme/gotch/nn"
// 	"github.com/sugarme/gotch/ts"
// 	"gonum.org/v1/plot"
// 	"gonum.org/v1/plot/plotter"
// 	"gonum.org/v1/plot/plotutil"
// 	"gonum.org/v1/plot/vg"
// )

// const lr = 1e-4
// const epoch = 1000
// const batchSize = 64
// const steps = 32
// const featureSize = 8
// const unitSize = steps * featureSize
// const hiddenSize = 32

// var vs = nn.NewVarStore(gotch.CPU)
// var points []float32

// func main() {
// 	i := 0.
// 	for {
// 		points = append(points, float32(math.Sin(i)))
// 		i += 0.001
// 		if i > 2*math.Pi {
// 			break
// 		}
// 	}

// 	loss := loss.NewMSE()
// 	optimizer := optimizer.NewAdam(optimizer.WithLearnRate(lr))
// 	m := newModel(loss, optimizer)

// 	p := plot.New()
// 	p.Title.Text = "predict sin(x)"
// 	p.X.Label.Text = "epoch"
// 	p.Y.Label.Text = "value"

// 	var real, predict plotter.XYs
// 	for i := 0; i < epoch; i++ {
// 		x, y, _ := getBatch(points, i+batchSize)
// 		loss := m.Train(i, x, y)
// 		x, _, ys := getBatch(points, i+batchSize)
// 		pred := m.Predict(x)
// 		y1 := ys[0]
// 		y2 := pred[0]
// 		real = append(real, plotter.XY{X: float64(i), Y: float64(y1)})
// 		predict = append(predict, plotter.XY{X: float64(i), Y: float64(y2)})
// 		if i%10 == 0 {
// 			acc := accuracy(m, i+batchSize)
// 			fmt.Printf("Epoch: %d, Loss: %e, Accuracy: %.02f%%\n", i, loss, acc)
// 			// fmt.Println(y.Value())
// 			// fmt.Println(pred.Value())
// 		}
// 	}

// 	lReal, err := plotter.NewLine(real)
// 	runtime.Assert(err)
// 	lReal.LineStyle.Color = plotutil.DarkColors[0]

// 	lPred, err := plotter.NewLine(predict)
// 	runtime.Assert(err)
// 	lPred.LineStyle.Color = plotutil.DarkColors[1]
// 	lPred.LineStyle.Dashes = []vg.Length{vg.Points(5), vg.Points(1)}

// 	p.Add(lReal, lPred)
// 	p.Y.Max = 1.5
// 	p.Legend.Add("real", lReal)
// 	p.Legend.Add("predict", lPred)
// 	p.Legend.Top = true
// 	p.Legend.XOffs = -20
// 	p.Save(16*vg.Inch, 4*vg.Inch, "sin.png")
// }

// func getBatch(points []float32, i int) (*ts.Tensor, *ts.Tensor, []float32) {
// 	x := make([]float32, batchSize*unitSize)
// 	y := make([]float32, batchSize)
// 	for batch := 0; batch < batchSize; batch++ {
// 		j := i + batch
// 		for t := 0; t < unitSize; t++ {
// 			x[batch*unitSize+t] = points[j%len(points)]
// 			j++
// 		}
// 		y[batch] = points[(i*batchSize+batch)%len(points)]
// 	}
// 	// rand.Shuffle(batchSize, func(i, j int) {
// 	// 	dx := make([]float32, unitSize)
// 	// 	dy := make([]float32, 1)
// 	// 	copy(dx, x[i*unitSize:(i+1)*unitSize])
// 	// 	copy(dy, y[i*1:(i+1)*1])
// 	// 	copy(x[i*unitSize:(i+1)*unitSize], x[j*unitSize:(j+1)*unitSize])
// 	// 	copy(y[i*1:(i+1)*1], y[j*1:(j+1)*1])
// 	// 	copy(x[j*unitSize:(j+1)*unitSize], dx)
// 	// 	copy(y[j*1:(j+1)*1], dy)
// 	// })
// 	xs, err := ts.NewTensorFromData(x, []int64{batchSize, steps, featureSize})
// 	runtime.Assert(err)
// 	ys, err := ts.NewTensorFromData(y, []int64{batchSize, 1})
// 	runtime.Assert(err)
// 	return xs, ys, y
// }

// func accuracy(m *model, i int) float32 {
// 	x, _, y := getBatch(points, i)
// 	pred := m.Predict(x)
// 	var correct float32
// 	for i := 0; i < batchSize; i++ {
// 		diff := 1 - float32(math.Abs(float64(y[i])-
// 			float64(pred[i])))
// 		if diff > 0 {
// 			correct += diff
// 		}
// 	}
// 	return correct * 100 / batchSize
// }
