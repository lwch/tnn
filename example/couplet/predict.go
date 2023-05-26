package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/model"
	"github.com/lwch/tnn/nn/tensor"
)

func predict(str string, vocabs []string, vocab2idx map[string]int, embedding [][]float64) {
	dir := filepath.Join(modelDir, "encoder.model")
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		panic("encoder model not found")
	}
	var encoder model.Model
	runtime.Assert(encoder.Load(dir))
	dir = filepath.Join(modelDir, "decoder.model")
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		panic("decoder model not found")
	}
	var decoder model.Model
	runtime.Assert(decoder.Load(dir))
	// var pred string
	// mix := embedding[rand.Intn(len(embedding))]
	// y := tensor.New(mix, 1, embeddingDim)
	// for _, ch := range str {
	// x := tensor.New(embedding[vocab2idx[string(ch)]], 1, embeddingDim)
	// for _, layer := range encoder.Layers() {
	// 	x = layer.Forward(x, false)
	// }
	// layers := decoder.Layers()
	// y = layers[0].Forward(y, false)
	// y = layers[1].Forward(y, false)
	// y = layers[2].(*layer.SelfAttention).ForwardQKV(y, x, y, false)
	// for i := 3; i < len(layers); i++ {
	// 	y = layers[i].Forward(y, false)
	// }
	// y = tensor.FromDense(y.Value())
	// pred += tensorStr(x, vocabs, embedding)
	// }
	dx := make([]float64, 0, unitSize)
	dy := make([]float64, 0, unitSize)
	for _, ch := range str {
		dx = append(dx, embedding[vocab2idx[string(ch)]]...)
		dy = append(dy, embedding[rand.Intn(len(embedding))]...)
	}
	for len(dx) < unitSize {
		dx = append(dx, embedding[1]...)
		dy = append(dy, embedding[1]...)
	}
	x := tensor.New(dx, 1, unitSize)
	y := tensor.New(dy, 1, unitSize)
	for _, layer := range encoder.Layers() {
		x = layer.Forward(x, false)
	}
	layers := decoder.Layers()
	y = layers[0].Forward(y, false)
	y = layers[1].Forward(y, false)
	y = layers[2].(*layer.SelfAttention).ForwardQKV(y, x, y, false)
	for i := 3; i < len(layers); i++ {
		y = layers[i].Forward(y, false)
	}
	fmt.Println(tensorStr(y, vocabs, embedding))
}

func lookupEmbedding(embedding [][]float64, v []float64) int {
	// fmt.Println(v)
	min := math.MaxFloat64
	ret := 0
	for i := 0; i < len(embedding); i++ {
		d := 0.0
		for j := 0; j < len(embedding[i]); j++ {
			d += math.Pow(embedding[i][j]-v[j], 2)
		}
		if d < min {
			min = d
			ret = i
		}
	}
	return ret
}

func tensorStr(x *tensor.Tensor, vocabs []string, embedding [][]float64) []string {
	rows, _ := x.Dims()
	var ret []string
	for i := 0; i < rows; i++ {
		var str string
		for j := 0; j < padSize; j++ {
			v := make([]float64, embeddingDim)
			for j := 0; j < embeddingDim; j++ {
				v[j] = x.Value().At(i, j*embeddingDim+j)
			}
			idx := lookupEmbedding(embedding, v)
			if idx == 1 {
				break
			}
			str += vocabs[idx]
		}
		ret = append(ret, str)
	}
	return ret
}
