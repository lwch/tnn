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
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/model"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
)

const modelDir = "./model"
const embeddingDim = 8 // 8个float64表示一个字向量
const unitSize = padSize * embeddingDim
const batchSize = 32
const epoch = 1000
const lr = 1e-3

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
	loss := loss.NewSoftmax()
	optimizer := optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)

	ch := make(chan []int)
	var wg sync.WaitGroup
	wg.Add(rt.NumCPU())
	var cnt atomic.Uint64
	for i := 0; i < rt.NumCPU(); i++ {
		go func() {
			defer wg.Done()
			trainWorker(loss, optimizer, trainX, trainY, embedding, ch, &cnt)
		}()
	}
	go showProgress(&cnt, len(trainX))

	begin := time.Now()
	for i := 0; i < epoch; i++ {
		cnt.Store(0)
		trainEpoch(trainX, trainY, embedding, ch)
		if i%10 == 0 {
			fmt.Printf("cost=%s, loss=%.05f\n", time.Since(begin).String(),
				avgLoss(loss, trainX, trainY, embedding))
		}
	}
	close(ch)
	wg.Wait()
	save()
}

func showProgress(cnt *atomic.Uint64, total int) {
	tk := time.NewTicker(time.Second)
	defer tk.Stop()
	upd := time.Now()
	for {
		<-tk.C
		fmt.Printf("train: %d/%d\r", cnt.Load(), total)
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
		pred := forward(x, y)
		grad := loss.Loss(pred, labels(y))
		grad.ZeroGrad()
		grad.Backward(grad)
		params := getParams()
		optimizer.Update(params)
		cnt.Add(uint64(len(idx)))
	}
}

func labels(y *tensor.Tensor) *tensor.Tensor {
	return y.Scale(0.9)
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
	idx := make([]int, len(trainX))
	for i := range trainX {
		idx[i] = i
	}
	rand.Shuffle(len(idx), func(i, j int) {
		idx[i], idx[j] = idx[j], idx[i]
	})
	var sum float64
	xIn := make([][]int, 0, batchSize)
	xOut := make([][]int, 0, batchSize)
	for cnt, i := range idx {
		if cnt >= 10000 {
			break
		}
		xIn = append(xIn, trainX[i])
		xOut = append(xOut, trainY[i])
		if len(xIn) < batchSize {
			continue
		}
		x, y := buildTensor(xIn, xOut, embedding)
		pred := forward(x, y)
		loss := loss.Loss(pred, labels(y)).Value()
		sum += loss.At(0, 0)
		xIn = xIn[:0]
		xOut = xOut[:0]
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
	init := initializer.NewXavierUniform(1)
	encoder = append(encoder, layer.NewSelfAttention(unitSize, init))
	encoder = append(encoder, layer.NewNor())
	encoder = append(encoder, layer.NewDense(unitSize*4, init))
	encoder = append(encoder, activation.NewReLU())
	encoder = append(encoder, layer.NewDense(unitSize, init))
	encoder = append(encoder, layer.NewNor())

	decoder = append(decoder, layer.NewSelfAttention(unitSize, init))
	decoder = append(decoder, layer.NewNor())
	decoder = append(decoder, layer.NewSelfAttention(unitSize, init))
	decoder = append(decoder, layer.NewNor())
	decoder = append(decoder, layer.NewDense(unitSize*4, init))
	decoder = append(decoder, activation.NewReLU())
	decoder = append(decoder, layer.NewDense(unitSize, init))
	decoder = append(decoder, layer.NewNor())
}

func forward(x, y *tensor.Tensor) *tensor.Tensor {
	for i := range encoder {
		x = encoder[i].Forward(x, true)
	}
	y = decoder[0].Forward(y, true)
	y = decoder[1].Forward(y, true)
	y = decoder[2].(*layer.SelfAttention).ForwardQKV(y, x, y, true)
	for i := 3; i < len(decoder); i++ {
		y = decoder[i].Forward(y, true)
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
