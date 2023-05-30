package main

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	rt "runtime"
	"sync"
	"sync/atomic"
	"time"

	_ "net/http/pprof"

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
const embeddingDim = 2 // 16个float64表示一个字向量
const unitSize = paddingSize * embeddingDim
const head = 1
const batchSize = 8
const epoch = 1000
const lr = 0.01
const transformerSize = 1

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

func train(trainX, trainY [][]int, vocabs []string, embedding [][]float64) {
	go func() { // for pprof
		http.ListenAndServe(":8888", nil)
	}()
	initModel(len(embedding))
	loss := loss.NewSoftmax()
	// loss := loss.NewMSE()
	optimizer := optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)
	// optimizer := optimizer.NewSGD(lr, 0)

	var cnt atomic.Uint64
	go showProgress(&cnt, len(trainX))

	begin := time.Now()
	i := 0
	for {
		cnt.Store(0)
		trainEpoch(&cnt, loss, optimizer, trainX, trainY, vocabs, embedding)
		if i%10 == 0 {
			list := getParams()
			for i, params := range list {
				var cnt int
				params.Range(func(_ string, p *tensor.Tensor) {
					rows, cols := p.Dims()
					cnt += rows * cols
				})
				fmt.Printf("layer %d params: %d\n", i, cnt)
			}
			fmt.Printf("cost=%s, loss=%.05f\n",
				time.Since(begin).String(),
				avgLoss(loss, trainX, trainY, vocabs, embedding))
		}
		i++
	}
	save()
}

func showProgress(cnt *atomic.Uint64, total int) {
	tk := time.NewTicker(time.Second)
	defer tk.Stop()
	upd := time.Now()
	for {
		<-tk.C
		fmt.Printf("train: %d/%d\r", cnt.Load(), total)
		if time.Since(upd).Seconds() >= 10 { // 每隔10秒保存一次模型
			save()
			upd = time.Now()
		}
	}
}

func dup(x []int) []int {
	ret := make([]int, len(x))
	copy(ret, x)
	return ret
}

func trainWorker(loss loss.Loss, trainX, trainY [][]int,
	vocabs []string, embedding [][]float64, ch chan []int, cnt *atomic.Uint64) {
	for {
		idx, ok := <-ch
		if !ok {
			return
		}
		xIn := make([][]int, 0, batchSize)
		xOut := make([][]int, 0, batchSize)
		for _, i := range idx {
			xIn = append(xIn, dup(trainX[i]))
			xOut = append(xOut, dup(trainY[i]))
		}
		x, y, z := buildTensor(xIn, xOut, vocabs, embedding, true)
		pred := forward(x, y, true)
		grad := loss.Loss(pred, z)
		grad.Backward(grad)
		cnt.Add(uint64(len(idx)))
	}
}

func trainEpoch(cnt *atomic.Uint64, loss loss.Loss, optimizer optimizer.Optimizer,
	trainX, trainY [][]int, vocabs []string, embedding [][]float64) {
	idx := make([]int, len(trainX))
	for i := range trainX {
		idx[i] = i
	}
	rand.Shuffle(len(idx), func(i, j int) {
		idx[i], idx[j] = idx[j], idx[i]
	})

	workerCount := rt.NumCPU()
	workerCount = 1

	ch := make(chan []int)
	var wg sync.WaitGroup
	wg.Add(workerCount)
	for i := 0; i < workerCount; i++ {
		go func() {
			defer wg.Done()
			trainWorker(loss, trainX, trainY, vocabs, embedding, ch, cnt)
		}()
	}

	for i := 0; i < len(trainX); i += batchSize {
		list := make([]int, 0, batchSize)
		for j := 0; j < batchSize; j++ {
			if i+j >= len(trainX) {
				break
			}
			list = append(list, idx[i+j])
		}
		ch <- list
	}
	close(ch)
	wg.Wait()
	params := getParams()
	optimizer.Update(params)
	zeroGrads(params)
}

var sumLoss float64

func lossWorker(loss loss.Loss, trainX, trainY [][]int, vocabs []string, embedding [][]float64, ch chan []int) {
	for {
		idx, ok := <-ch
		if !ok {
			return
		}
		xIn := make([][]int, 0, batchSize)
		xOut := make([][]int, 0, batchSize)
		for _, i := range idx {
			xIn = append(xIn, dup(trainX[i]))
			xOut = append(xOut, dup(trainY[i]))
		}
		x, y, z := buildTensor(xIn, xOut, vocabs, embedding, true)
		pred := forward(x, y, false)
		loss := loss.Loss(pred, z).Value()
		sumLoss += loss.At(0, 0)
	}
}

func avgLoss(loss loss.Loss, trainX, trainY [][]int, vocabs []string, embedding [][]float64) float64 {
	sumLoss = 0

	ch := make(chan []int)
	var wg sync.WaitGroup
	wg.Add(rt.NumCPU())
	for i := 0; i < rt.NumCPU(); i++ {
		go func() {
			defer wg.Done()
			lossWorker(loss, trainX, trainY, vocabs, embedding, ch)
		}()
	}

	var size float64
	var list []int
	for i := range trainX {
		list = append(list, i)
		if len(list) < batchSize {
			continue
		}
		dup := make([]int, len(list))
		copy(dup, list)
		ch <- dup
		list = list[:0]
		size++
	}
	if len(list) > 0 {
		ch <- list
		size++
	}
	close(ch)
	wg.Wait()
	return sumLoss / size
}

var layers []layer.Layer

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

func zeroGrads(paramList []*params.Params) {
	for _, params := range paramList {
		params.Range(func(_ string, p *tensor.Tensor) {
			p.ZeroGrad()
		})
	}
}

func addTransformer(init initializer.Initializer) {
	layers = append(layers, layer.NewSelfAttention(paddingSize, embeddingDim, head, init))
	// layers = append(layers, layer.NewNor())
	layers = append(layers, layer.NewDense(unitSize*4, init))
	layers = append(layers, activation.NewReLU())
	layers = append(layers, layer.NewDense(unitSize, init))
	// layers = append(layers, layer.NewNor())
}

func initModel(vocabSize int) {
	init := initializer.NewXavierUniform(1)
	for i := 0; i < transformerSize; i++ {
		addTransformer(init)
	}
	layers = append(layers, activation.NewReLU())
	layers = append(layers, layer.NewDense(vocabSize, init))
}

var dropout = layer.NewDropout(0.5)

func forwardTransformer(i int, x, y *tensor.Tensor, train bool) (*tensor.Tensor, int) {
	srcY := y
	y = layers[i].(*layer.SelfAttention).ForwardQKV(x, y, y, true, train)
	selfOut := y.Add(srcY)
	// if train {
	// 	y = dropout.Forward(y, true)
	// }
	// selfOut := layers[i+1].Forward(y, train) // nor
	// y = layers[i+1].Forward(y, train) // nor
	y = layers[i+1].Forward(y, train) // dense
	y = layers[i+2].Forward(y, train) // relu
	y = layers[i+3].Forward(y, train) // dense
	y = y.Add(selfOut)
	// if train {
	// 	y = dropout.Forward(y, true)
	// }
	// y = layers[i+5].Forward(y, train) // nor
	return y, i + 4
}

func forward(x, y *tensor.Tensor, train bool) *tensor.Tensor {
	i := 0
	for j := 0; j < transformerSize; j++ {
		y, i = forwardTransformer(i, x, y, train)
	}
	y = layers[i].Forward(y, train)   // relu
	y = layers[i+1].Forward(y, train) // output
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
	saveModel(layers, "couplet")
	fmt.Println("model saved")
}
