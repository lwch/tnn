package main

import (
	"fmt"
	"math"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/optimizer"
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

const lr = 0.001
const epoch = 1000
const batchSize = 16
const seqSize = 8
const dims = 8
const heads = 4
const unitSize = seqSize * dims
const transformerSize = 1

func main() {
	// f, err := os.Create("cpu.pprof")
	// runtime.Assert(err)
	// defer f.Close()
	// runtime.Assert(pprof.StartCPUProfile(f))
	// go func() {
	// 	time.Sleep(time.Minute)
	// 	pprof.StopCPUProfile()
	// 	f.Close()
	// 	os.Exit(0)
	// }()
	// defer pprof.StopCPUProfile()

	loss := loss.NewMSE()
	optimizer := optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)

	p := plot.New()
	p.Title.Text = "predict sin(x)"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "value"

	var real, predict plotter.XYs
	var points []float64
	i := 0.
	for {
		points = append(points, math.Sin(i))
		i += 0.001
		if i > 2*math.Pi {
			break
		}
	}
	for i := 0; i < epoch; i++ {
		input, output := getBatch(points, i+batchSize)
		pred := forward(input, true)
		grad := loss.Loss(pred, output)
		grad.ZeroGrad()
		grad.Backward(grad)
		paramList := getParams()
		optimizer.Update(paramList)
		if i%10 == 0 {
			pred = forward(input, false)
			acc := accuracy(pred, output)
			for j := 0; j < batchSize; j++ {
				real = append(real, plotter.XY{X: float64(i*batchSize + j), Y: output.Value().At(j, 0)})
				predict = append(predict, plotter.XY{X: float64(i*batchSize + j), Y: pred.Value().At(j, 0)})
			}
			loss := loss.Loss(pred, output)
			fmt.Printf("Epoch: %d, Loss: %.05f, Accuracy: %.02f%%\n",
				i, loss.Value().At(0, 0), acc)
			// fmt.Println(mat.Formatted(output.Value()))
			// fmt.Println(mat.Formatted(pred.Value()))
		}
	}

	lReal, err := plotter.NewLine(real)
	runtime.Assert(err)
	lReal.LineStyle.Color = plotutil.DarkColors[0]

	lPred, err := plotter.NewLine(predict)
	runtime.Assert(err)
	lPred.LineStyle.Color = plotutil.DarkColors[1]
	lPred.LineStyle.Dashes = []vg.Length{vg.Points(10), vg.Points(5)}

	p.Add(lReal, lPred)
	p.Y.Max = 1.5
	p.Legend.Add("real", lReal)
	p.Legend.Add("predict", lPred)
	p.Legend.Top = true
	p.Legend.XOffs = -20
	p.Save(8*vg.Inch, 4*vg.Inch, "sin.png")
}

var layers []layer.Layer

func addTransformer(init initializer.Initializer) {
	layers = append(layers, layer.NewSelfAttention(seqSize, dims, heads, init))
	layers = append(layers, layer.NewNor())
	layers = append(layers, layer.NewDense(unitSize*4, init))
	layers = append(layers, activation.NewSigmoid())
	layers = append(layers, layer.NewDense(unitSize, init))
	layers = append(layers, layer.NewNor())
}

func init() {
	init := initializer.NewXavierUniform(1)
	// transformer层
	for i := 0; i < transformerSize; i++ {
		addTransformer(init)
	}
	// 输出层
	layers = append(layers, activation.NewSigmoid())
	layers = append(layers, layer.NewDense(1, init))
}

func forwardTransformer(i int, x *tensor.Tensor, train bool) (*tensor.Tensor, int) {
	// y := layers[i].Forward(x, train) // self attention
	y := layers[i].(*layer.SelfAttention).ForwardQKV(x, x, x, true, train) // self attention
	y = y.Add(x)
	selfOut := layers[i+1].Forward(y, train) // nor
	y = layers[i+2].Forward(selfOut, train)  // dense
	y = layers[i+3].Forward(y, train)        // relu
	y = layers[i+4].Forward(y, train)        // dense
	y = y.Add(selfOut)
	y = layers[i+5].Forward(y, train) // nor
	return y, i + 6
}

func forward(x *tensor.Tensor, train bool) *tensor.Tensor {
	y := x
	i := 0
	for j := 0; j < transformerSize; j++ {
		y, i = forwardTransformer(i, y, train)
	}
	y = layers[i].Forward(y, train)   // relu
	y = layers[i+1].Forward(y, train) // output
	return y
}

func getParams() []*params.Params {
	var ret []*params.Params
	for _, layer := range layers {
		params := layer.Params()
		if params.IsEmpty() {
			continue
		}
		ret = append(ret, params)
	}
	return ret
}

func getBatch(points []float64, i int) (*tensor.Tensor, *tensor.Tensor) {
	x := make([]float64, batchSize*unitSize)
	y := make([]float64, batchSize)
	for batch := 0; batch < batchSize; batch++ {
		j := i + batch
		for t := 0; t < unitSize; t++ {
			x[batch*unitSize+t] = points[j%len(points)]
			j++
		}
		y[batch] = points[(i*batchSize+batch)%len(points)]
	}
	// rand.Shuffle(batchSize, func(i, j int) {
	// 	dx := make([]float64, unitSize)
	// 	dy := make([]float64, 1)
	// 	copy(dx, x[i*unitSize:(i+1)*unitSize])
	// 	copy(dy, y[i*1:(i+1)*1])
	// 	copy(x[i*unitSize:(i+1)*unitSize], x[j*unitSize:(j+1)*unitSize])
	// 	copy(y[i*1:(i+1)*1], y[j*1:(j+1)*1])
	// 	copy(x[j*unitSize:(j+1)*unitSize], dx)
	// 	copy(y[j*1:(j+1)*1], dy)
	// })
	return tensor.New(x, batchSize, unitSize),
		tensor.New(y, batchSize, 1)
}

func accuracy(pred, output *tensor.Tensor) float64 {
	var correct float64
	for i := 0; i < batchSize; i++ {
		diff := 1 - math.Abs(output.Value().At(i, 0)-pred.Value().At(i, 0))
		if diff > 0 {
			correct += diff
		}
	}
	return correct * 100 / batchSize
}
