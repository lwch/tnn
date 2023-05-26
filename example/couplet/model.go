package main

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	rt "runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/internal/math"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/model"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
)

const modelDir = "./model"
const embeddingDim = 2 // 8个float64表示一个字向量
const unitSize = padSize * embeddingDim
const batchSize = 16
const epoch = 10
const lr = 1e-3

var testX, testY *tensor.Tensor

func buildEmbedding(vocabSize int) {
	init := initializer.NewXavierUniform(1)
	data := init.RandShape(vocabSize, embeddingDim)
	os.MkdirAll(modelDir, 0755)
	dir := filepath.Join(modelDir, "embedding")
	f, err := os.Create(dir)
	runtime.Assert(err)
	defer f.Close()
	runtime.Assert(binary.Write(f, binary.BigEndian, data))
}

func loadEmbedding(vocabSize int) [][]float64 {
	dir := filepath.Join(modelDir, "embedding")
	f, err := os.Open(dir)
	runtime.Assert(err)
	defer f.Close()
	var ret [][]float64
	for i := 0; i < vocabSize; i++ {
		data := make([]float64, embeddingDim)
		runtime.Assert(binary.Read(f, binary.BigEndian, &data))
		ret = append(ret, data)
	}
	return ret
}

func train(trainX, trainY [][]int, embedding [][]float64) {
	loss := loss.NewMSE()
	optimizer := optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)

	ch := make(chan []int)
	var wg sync.WaitGroup
	wg.Add(rt.NumCPU())
	var cnt atomic.Uint64
	var begin time.Time
	for i := 0; i < rt.NumCPU(); i++ {
		go func() {
			defer wg.Done()
			trainWorker(loss, optimizer, trainX, trainY, embedding, ch, &cnt)
		}()
	}
	go showProgress(loss, &begin, &cnt, len(trainX))

	for i := 0; i < epoch; i++ {
		cnt.Store(0)
		begin = time.Now()
		trainEpoch(trainX, trainY, embedding, ch)
		fmt.Printf("loss=%.05f\n", avgLoss(loss, trainX, trainY, embedding))
	}
	close(ch)
	wg.Wait()
	save()
}

func showProgress(loss loss.Loss, begin *time.Time, cnt *atomic.Uint64, total int) {
	tk := time.NewTicker(time.Second)
	defer tk.Stop()
	upd := time.Now()
	for {
		<-tk.C
		// pred := testX
		// y := testY
		// for _, layer := range encoder {
		// 	pred = layer.Forward(pred, false)
		// }
		// loss := math.Softmax(pred, 1).Sub(y).Sum().Value()
		// loss := loss.Loss(pred, y).Value()
		fmt.Printf("train: %d/%d, cost=%s\n", cnt.Load(),
			total, time.Since(*begin).String())
		// fmt.Println(mat.Formatted(loss))
		if time.Since(upd).Seconds() >= 60 { // 每隔1分钟保存一次模型
			save()
			upd = time.Now()
		}
	}
}

func trainWorker(loss loss.Loss, optimizer optimizer.Optimizer,
	trainX, trainY [][]int, embedding [][]float64, ch chan []int, cnt *atomic.Uint64) {
	for {
		idx, ok := <-ch
		if !ok {
			return
		}
		xIn := make([][]int, 0, batchSize)
		xOut := make([][]int, 0, batchSize)
		for _, i := range idx {
			xIn = append(xIn, trainX[i])
			xOut = append(xOut, trainY[i])
		}
		rand.Shuffle(len(xIn), func(i, j int) {
			xIn[i], xIn[j] = xIn[j], xIn[i]
			xOut[i], xOut[j] = xOut[j], xOut[i]
		})
		x, y := buildTensor(xIn, xOut, embedding)
		testX, testY = x, y
		pred := forward(x, y)
		pred = math.Softmax(pred, 1)
		ones := tensor.Ones(pred.Dims())
		grad := ones.Sub(pred).Sum().Scale(1 / float64(len(idx)))
		// grad := math.Softmax(pred, 1).Log().Sub(ones).Sum()
		// grad := loss.Loss(pred, y)
		grad.ZeroGrad()
		grad.Backward(grad)
		params := getParams()
		optimizer.Update(params)
		// test(x)
		cnt.Add(uint64(len(idx)))
	}
}

func test(x *tensor.Tensor) {
	for _, layer := range encoder {
		x = layer.Forward(x, false)
	}
	fmt.Println(mat.Formatted(x.Value()))
}

func trainEpoch(trainX, trainY [][]int, embedding [][]float64, ch chan []int) {
	for i := 0; i < len(trainX); i += batchSize {
		list := make([]int, 0, batchSize)
		for j := 0; j < batchSize; j++ {
			if i+j >= len(trainX) {
				break
			}
			list = append(list, i)
		}
		ch <- list
	}
}

func avgLoss(loss loss.Loss, trainX, trainY [][]int, embedding [][]float64) float64 {
	var sum float64
	for i := 0; i < len(trainX); i += batchSize {
		xIn := make([][]int, 0, batchSize)
		xOut := make([][]int, 0, batchSize)
		for j := 0; j < batchSize; j++ {
			if i+j >= len(trainX) {
				break
			}
			xIn = append(xIn, trainX[i+j])
			xOut = append(xOut, trainY[i+j])
		}
		x, y := buildTensor(xIn, xOut, embedding)
		pred := forward(x, y)
		pred = math.Softmax(pred, 1)
		ones := tensor.Ones(pred.Dims())
		// loss := math.Softmax(pred, 1).Sum().Value()
		loss := ones.Sub(pred).Sum().Value()
		sum += loss.At(0, 0)
	}
	return sum / float64(len(trainX))
}

var encoder []layer.Layer
var decoder []layer.Layer

func getParams() []*params.Params {
	var ret []*params.Params
	for _, layer := range encoder {
		params := layer.Params()
		if params.IsEmpty() {
			continue
		}
		ret = append(ret, params)
	}
	for _, layer := range decoder {
		params := layer.Params()
		if params.IsEmpty() {
			continue
		}
		ret = append(ret, params)
	}
	return ret
}

func init() {
	// init := initializer.NewXavierUniform(1)
	init := initializer.NewZero()
	encoder = append(encoder, layer.NewSelfAttention(unitSize, init))
	encoder = append(encoder, layer.NewNor())
	encoder = append(encoder, layer.NewDense(unitSize*4, init))
	// encoder = append(encoder, activation.NewReLU())
	encoder = append(encoder, layer.NewDense(unitSize, init))
	encoder = append(encoder, layer.NewNor())

	decoder = append(decoder, layer.NewSelfAttention(unitSize, init))
	decoder = append(decoder, layer.NewNor())
	decoder = append(decoder, layer.NewSelfAttention(unitSize, init))
	decoder = append(decoder, layer.NewNor())
	decoder = append(decoder, layer.NewDense(unitSize*4, init))
	// decoder = append(decoder, activation.NewReLU())
	decoder = append(decoder, layer.NewDense(unitSize, init))
	decoder = append(decoder, layer.NewNor())
}

// func haveNan(x *tensor.Tensor, prefix string) {
// 	rows, cols := x.Dims()
// 	for i := 0; i < rows; i++ {
// 		for j := 0; j < cols; j++ {
// 			if math.IsNaN(x.Value().At(i, j)) {
// 				fmt.Println(prefix, "!!!!!!!!!!")
// 				return
// 			}
// 		}
// 	}
// }

func forward(x, y *tensor.Tensor) *tensor.Tensor {
	// haveNan(x, "x")
	// haveNan(y, "y")
	for i := range encoder {
		x = encoder[i].Forward(x, true)
		// haveNan(x, fmt.Sprintf("encoder:%d", i))
	}
	y = decoder[0].Forward(y, true)
	// haveNan(y, fmt.Sprintf("decoder:%d", 0))
	y = decoder[1].Forward(y, true)
	// haveNan(y, fmt.Sprintf("decoder:%d", 1))
	y = decoder[2].(*layer.SelfAttention).ForwardQKV(y, x, y, true)
	// haveNan(y, fmt.Sprintf("decoder:%d", 2))
	for i := 3; i < len(decoder); i++ {
		y = decoder[i].Forward(y, true)
		// haveNan(y, fmt.Sprintf("decoder:%d", i))
	}
	return y
}

func saveModel(layers []layer.Layer, name string) {
	var net net.Net
	net.Set(layers...)
	err := model.New(&net, loss.NewMSE(),
		optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)).
		Save(filepath.Join(modelDir, name+".model"))
	runtime.Assert(err)
}

func save() {
	saveModel(encoder, "encoder")
	saveModel(decoder, "decoder")
	fmt.Println("model saved")
}
