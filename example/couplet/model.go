package main

import (
	"encoding/binary"
	"fmt"
	"math"
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
const embeddingDim = 8 // 8个float64表示一个字向量
const unitSize = paddingSize * embeddingDim
const head = 2
const batchSize = 16
const epoch = 1000
const lr = 0.001
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
	// loss := loss.NewSoftmax()
	loss := loss.NewMSE()
	optimizer := optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)
	// optimizer := optimizer.NewSGD(lr, 0)

	var cnt atomic.Uint64
	go showProgress(&cnt, len(trainY)*paddingSize)

	begin := time.Now()
	i := 0
	for {
		cnt.Store(0)
		trainEpoch(&cnt, loss, optimizer, trainX, trainY, vocabs, embedding)
		if i%10 == 0 {
			list, names := getParams()
			for i, params := range list {
				var cnt int
				params.Range(func(_ string, p *tensor.Tensor) {
					rows, cols := p.Dims()
					cnt += rows * cols
				})
				fmt.Printf("%s: %d\n", names[i], cnt)
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

func trainWorker(loss loss.Loss, trainX, trainY [][]int,
	vocabs []string, embedding [][]float64, ch chan []int, cnt *atomic.Uint64) {
	for {
		idx, ok := <-ch
		if !ok {
			return
		}
		x := make([]float64, 0, len(idx)*unitSize)
		y := make([]float64, 0, len(idx)*embeddingDim)
		paddingMask := make([][]bool, 0, batchSize)
		for _, idx := range idx {
			i := math.Floor(float64(idx) / float64(paddingSize))
			j := idx % paddingSize
			dx := trainX[int(i)]
			dy := trainY[int(i)]
			dy = append([]int{0}, dy...) // <s> ...
			var dz int
			if j < len(dy) {
				dz = dy[j]
				dy = dy[:j]
			} else {
				dz = paddingIdx
			}
			xTrain, zTrain, pm := build(dx, dy, dz, vocabs, embedding)
			x = append(x, xTrain...)
			y = append(y, zTrain...)
			paddingMask = append(paddingMask, pm)
		}
		// xIn := make([][]int, 0, batchSize)
		// xOut := make([][]int, 0, batchSize)
		// for _, i := range idx {
		// 	xIn = append(xIn, dup(trainX[i]))
		// 	xOut = append(xOut, dup(trainY[i]))
		// }
		// x, y, z := buildTensor(xIn, xOut, vocabs, embedding, true)
		xIn := tensor.New(x, len(idx), unitSize)
		zOut := tensor.New(y, len(idx), len(embedding))
		pred := forward(xIn, buildPaddingMasks(paddingMask), true)
		grad := loss.Loss(pred, zOut)
		grad.Backward(grad)
		cnt.Add(uint64(len(idx)))
	}
}

func trainEpoch(cnt *atomic.Uint64, loss loss.Loss, optimizer optimizer.Optimizer,
	trainX, trainY [][]int, vocabs []string, embedding [][]float64) {
	idx := make([]int, len(trainY)*paddingSize)
	for i := 0; i < len(idx); i++ {
		idx[i] = i
	}
	rand.Shuffle(len(idx), func(i, j int) {
		idx[i], idx[j] = idx[j], idx[i]
	})

	workerCount := rt.NumCPU()
	// workerCount = 1

	ch := make(chan []int)
	var wg sync.WaitGroup
	wg.Add(workerCount)
	for i := 0; i < workerCount; i++ {
		go func() {
			defer wg.Done()
			trainWorker(loss, trainX, trainY, vocabs, embedding, ch, cnt)
		}()
	}

	for i := 0; i < len(idx); i += batchSize {
		list := make([]int, 0, batchSize)
		for j := 0; j < batchSize; j++ {
			if i+j >= len(idx) {
				break
			}
			list = append(list, idx[i+j])
		}
		ch <- list
	}
	close(ch)
	wg.Wait()
	params, _ := getParams()
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
		x := make([]float64, 0, len(idx)*unitSize)
		y := make([]float64, 0, len(idx)*embeddingDim)
		paddingMask := make([][]bool, 0, batchSize)
		for _, idx := range idx {
			i := math.Floor(float64(idx) / float64(paddingSize))
			j := idx % paddingSize
			dx := trainX[int(i)]
			dy := trainY[int(i)]
			dy = append([]int{0}, dy...) // <s> ...
			var dz int
			if j < len(dy) {
				dz = dy[j]
				dy = dy[:j]
			} else {
				dz = paddingIdx
			}
			xTrain, zTrain, pm := build(dx, dy, dz, vocabs, embedding)
			x = append(x, xTrain...)
			y = append(y, zTrain...)
			paddingMask = append(paddingMask, pm)
		}
		xIn := tensor.New(x, len(idx), unitSize)
		zOut := tensor.New(y, len(idx), len(embedding))
		pred := forward(xIn, buildPaddingMasks(paddingMask), false)
		loss := loss.Loss(pred, zOut).Value()
		sumLoss += loss.At(0, 0)
	}
}

func avgLoss(loss loss.Loss, trainX, trainY [][]int, vocabs []string, embedding [][]float64) float64 {
	sumLoss = 0

	idx := make([]int, len(trainY)*paddingSize)
	for i := 0; i < len(idx); i++ {
		idx[i] = i
	}

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
	for _, i := range idx {
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

func getParams() ([]*params.Params, []string) {
	var ret []*params.Params
	var names []string
	for _, layer := range layers {
		params := layer.Params()
		if params.IsEmpty() {
			continue
		}
		ret = append(ret, params)
		names = append(names, layer.Name())
	}
	return ret, names
}

func zeroGrads(paramList []*params.Params) {
	for _, params := range paramList {
		params.Range(func(_ string, p *tensor.Tensor) {
			p.ZeroGrad()
		})
	}
}

func addTransformer(init initializer.Initializer, i int) {
	attention := layer.NewSelfAttention(paddingSize, embeddingDim, head, init)
	attention.SetName(fmt.Sprintf("transformer%d_attention", i))
	layers = append(layers, attention)
	layers = append(layers, layer.NewNor())
	dense1 := layer.NewDense(unitSize*4, init)
	dense1.SetName(fmt.Sprintf("transformer%d_dense1", i))
	layers = append(layers, dense1)
	layers = append(layers, activation.NewReLU())
	dense2 := layer.NewDense(unitSize, init)
	dense2.SetName(fmt.Sprintf("transformer%d_dense2", i))
	layers = append(layers, dense2)
	layers = append(layers, layer.NewNor())
}

func initModel(vocabSize int) {
	init := initializer.NewXavierUniform(1)
	for i := 0; i < transformerSize; i++ {
		addTransformer(init, i)
	}
	layers = append(layers, activation.NewReLU())
	output := layer.NewDense(vocabSize, init)
	output.SetName("output")
	layers = append(layers, output)
}

var dropout = layer.NewDropout(0.1)
var featureMask *tensor.Tensor

func init() {
	featureMask = tensor.New(nil, paddingSize, paddingSize)
	for i := 0; i < paddingSize; i++ {
		for j := i; j < paddingSize; j++ {
			featureMask.Set(i, j, -1e9)
		}
	}
}

func buildPaddingMasks(masks [][]bool) []*tensor.Tensor {
	ret := make([]*tensor.Tensor, 0, len(masks))
	for batch := 0; batch < len(masks); batch++ {
		size := len(masks[batch])
		mask := tensor.New(nil, size, size)
		for i, b := range masks[batch] {
			if !b {
				continue
			}
			// 十字mask，别的词跟他，他跟别的词都需要mask掉
			for j := 0; j < size; j++ {
				mask.Set(j, i, -1e9)
			}
		}
		ret = append(ret, mask)
	}
	return ret
}

func forwardTransformer(i int, x *tensor.Tensor, paddingMasks []*tensor.Tensor, train bool) (*tensor.Tensor, int) {
	masks := make([]*tensor.Tensor, len(paddingMasks))
	for i, m := range paddingMasks {
		masks[i] = m.Add(featureMask)
	}
	y := layers[i].(*layer.SelfAttention).ForwardQKV(x, x, x, masks, train)
	y = y.Add(x)
	// if train {
	// 	y = dropout.Forward(y, true)
	// }
	selfOut := layers[i+1].Forward(y, train) // nor
	y = layers[i+2].Forward(selfOut, train)  // dense
	y = layers[i+3].Forward(y, train)        // relu
	y = layers[i+4].Forward(y, train)        // dense
	y = y.Add(selfOut)
	// if train {
	// 	y = dropout.Forward(y, true)
	// }
	y = layers[i+5].Forward(y, train) // nor
	return y, i + 6
}

var positionEmbedding []float64

func init() {
	positionEmbedding = make([]float64, unitSize)
	for k := 0; k < paddingSize; k++ {
		start := k * embeddingDim
		for i := 0; i < embeddingDim/2; i++ {
			n := float64(k) / math.Pow(10000, 2*float64(i)/float64(embeddingDim))
			positionEmbedding[start+i*2] = math.Sin(n)
			positionEmbedding[start+i*2+1] = math.Cos(n)
		}
	}
}

func buildPositionEmbedding(batchSize int) *tensor.Tensor {
	data := make([]float64, batchSize*unitSize)
	for i := 0; i < batchSize; i++ {
		start := i * unitSize
		copy(data[start:start+unitSize], positionEmbedding)
	}
	return tensor.New(data, batchSize, unitSize)
}

func forward(x *tensor.Tensor, paddingMasks []*tensor.Tensor, train bool) *tensor.Tensor {
	batchSize, _ := x.Dims()
	x = x.Add(buildPositionEmbedding(batchSize))
	i := 0
	var y *tensor.Tensor
	for j := 0; j < transformerSize; j++ {
		y, i = forwardTransformer(i, x, paddingMasks, train)
	}
	y = layers[i].Forward(y, train)   // relu
	y = layers[i+1].Forward(y, train) // output
	y = y.Softmax(1)                  // softmax
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
