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
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/model"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
)

const modelDir = "./model"
const embeddingDim = 2 // 2个float64表示一个字向量
const batchSize = 4
const epoch = 10
const lr = 0.001
const unitSize = padSize * embeddingDim

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
	go showProgress(&begin, &cnt, len(trainX))

	for i := 0; i < epoch; i++ {
		cnt.Store(0)
		begin = time.Now()
		trainEpoch(trainX, trainY, embedding, ch)
	}
	close(ch)
	wg.Wait()
}

func showProgress(begin *time.Time, cnt *atomic.Uint64, total int) {
	tk := time.NewTicker(time.Second)
	defer tk.Stop()
	i := 0
	for {
		<-tk.C
		fmt.Printf("train: %d/%d, cost=%s\r", cnt.Load(),
			total, time.Since(*begin).String())
		i++
		if i%60 == 0 { // 每隔1分钟保存一次模型
			save()
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
		x, y := buildTensor(xIn, xOut, embedding, true)
		pred := forward(x, y)
		grad := loss.Loss(pred, y)
		grad.ZeroGrad()
		grad.Backward(grad)
		params := getParams()
		optimizer.Update(params)
		// paramSize := 0
		// for _, ps := range params {
		// 	ps.Range(func(_ string, dense *tensor.Tensor) {
		// 		rows, cols := dense.Dims()
		// 		paramSize += rows * cols
		// 	})
		// }
		// pred = forward(x, y)
		// fmt.Println()
		// fmt.Println(loss.Loss(pred, y).Value().At(0, 0), paramSize)
		cnt.Add(uint64(len(idx)))
	}
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
	encoder = append(encoder, layer.NewSelfAttention(unitSize, init))
	// encoder = append(encoder, layer.NewDense(unitSize*4, init))
	// encoder = append(encoder, activation.NewReLU())
	// encoder = append(encoder, layer.NewDense(unitSize, init))

	decoder = append(decoder, layer.NewSelfAttention(unitSize, init))
	decoder = append(decoder, layer.NewSelfAttention(unitSize, init))
	// decoder = append(decoder, layer.NewDense(unitSize*4, init))
	// decoder = append(decoder, activation.NewReLU())
	// decoder = append(decoder, layer.NewDense(unitSize, init))
}

func forward(x, y *tensor.Tensor) *tensor.Tensor {
	for i := range encoder {
		x = encoder[i].Forward(x, true)
	}
	y = decoder[0].Forward(y, true)
	y = decoder[1].(*layer.SelfAttention).ForwardQKV(y, x, y, true)
	for i := 2; i < len(decoder); i++ {
		y = decoder[i].Forward(y, true)
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
