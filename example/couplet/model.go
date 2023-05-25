package main

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/tensor"
)

const modelDir = "./model"
const embeddingDim = 4 // 4个float64表示一个字向量
const batchSize = 1
const epoch = 100
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
	idx := make([]int, len(trainX))
	for i := range idx {
		idx[i] = i
	}

	for i := 0; i < epoch; i++ {
		trainEpoch(trainX, trainY, embedding, idx)
	}
}

func trainEpoch(trainX, trainY [][]int, embedding [][]float64, idx []int) {
	rand.Shuffle(len(idx), func(i, j int) {
		idx[i], idx[j] = idx[j], idx[i]
	})
	for i := 0; i < len(idx); i += batchSize {
		xIn := make([][]int, 0, batchSize)
		xOut := make([][]int, 0, batchSize)
		for j := 0; j < batchSize; j++ {
			if i+j >= len(idx) {
				break
			}
			xIn = append(xIn, trainX[idx[i+j]])
			xOut = append(xOut, trainY[idx[i+j]])
		}
		x, y := buildTensor(xIn, xOut, embedding, true)
		y = forward(x, y)
		fmt.Println(y.Dims())
	}
}

var encoder []layer.Layer
var decoder []layer.Layer

func init() {
	init := initializer.NewXavierUniform(1)
	encoder = append(encoder, layer.NewSelfAttention(unitSize, init))
	encoder = append(encoder, layer.NewDense(unitSize*4, init))
	encoder = append(encoder, activation.NewReLU())
	encoder = append(encoder, layer.NewDense(unitSize, init))
	decoder = append(decoder, layer.NewSelfAttention(unitSize, init))
	decoder = append(decoder, layer.NewSelfAttention(unitSize, init))
	decoder = append(decoder, layer.NewDense(unitSize*4, init))
	decoder = append(decoder, activation.NewReLU())
	decoder = append(decoder, layer.NewDense(unitSize, init))
}

func forward(x, y *tensor.Tensor) *tensor.Tensor {
	for i := range encoder {
		x = encoder[i].Forward(x, true)
		rows, cols := x.Dims()
		fmt.Println("encoder", i, rows, cols)
	}
	y = decoder[0].Forward(y, true)
	rows, cols := y.Dims()
	fmt.Println("decoder", 0, rows, cols)
	y = decoder[1].(*layer.SelfAttention).ForwardQKV(y, x, y, true)
	rows, cols = y.Dims()
	fmt.Println("decoder", 1, rows, cols)
	for i := 2; i < len(decoder); i++ {
		y = decoder[i].Forward(y, true)
		rows, cols := y.Dims()
		fmt.Println("decoder", i, rows, cols)
	}
	return y
}
