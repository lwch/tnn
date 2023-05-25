package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	rt "runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/model"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
)

const modelDir = "./model"
const embeddingDim = 32 // 32个float64表示一个字向量
const batchSize = 32
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
		// grad := math.Softmax(pred.Sub(y), 1).Sum()
		grad := loss.Loss(pred, y)
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
	init := initializer.NewXavierUniform(1)
	encoder = append(encoder, layer.NewSelfAttention(embeddingDim, init))
	encoder = append(encoder, layer.NewNor())
	encoder = append(encoder, layer.NewDense(embeddingDim*4, init))
	encoder = append(encoder, activation.NewReLU())
	encoder = append(encoder, layer.NewDense(embeddingDim, init))
	encoder = append(encoder, layer.NewNor())

	decoder = append(decoder, layer.NewSelfAttention(embeddingDim, init))
	decoder = append(decoder, layer.NewNor())
	decoder = append(decoder, layer.NewSelfAttention(embeddingDim, init))
	decoder = append(decoder, layer.NewNor())
	decoder = append(decoder, layer.NewDense(embeddingDim*4, init))
	decoder = append(decoder, activation.NewReLU())
	decoder = append(decoder, layer.NewDense(embeddingDim, init))
	decoder = append(decoder, layer.NewNor())
}

func haveNan(x *tensor.Tensor, prefix string) {
	rows, cols := x.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if math.IsNaN(x.Value().At(i, j)) {
				fmt.Println(prefix, "!!!!!!!!!!")
				return
			}
		}
	}
}

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
	for i := 2; i < len(decoder); i++ {
		y = decoder[i].Forward(y, true)
		// haveNan(y, fmt.Sprintf("decoder:%d", i))
	}
	return y
}

func save() {
	var net net.Net
	for _, layer := range encoder {
		net.Add(layer)
	}
	dir := filepath.Join(modelDir, "encoder.model")
	err := model.New(&net, loss.NewMSE(),
		optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)).Save(dir)
	runtime.Assert(err)
}
