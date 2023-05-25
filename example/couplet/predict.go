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
	var pred string
	var y *tensor.Tensor
	for _, ch := range str {
		x := tensor.New(embedding[vocab2idx[string(ch)]], 1, embeddingDim)
		for _, layer := range encoder.Layers() {
			x = layer.Forward(x, false)
		}
		layers := decoder.Layers()
		if y == nil {
			mix := embedding[rand.Intn(len(embedding))]
			y = tensor.New(mix, 1, embeddingDim)
		}
		y = layers[0].Forward(y, false)
		y = layers[1].Forward(y, false)
		y = layers[2].(*layer.SelfAttention).ForwardQKV(y, x, y, false)
		for i := 3; i < len(layers); i++ {
			y = layers[i].Forward(y, false)
		}
		pred += tensorStr(y, vocabs, embedding)
	}
	fmt.Println(pred)
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

func tensorStr(x *tensor.Tensor, vocabs []string, embedding [][]float64) string {
	rows, _ := x.Dims()
	var str string
	for i := 0; i < rows; i++ {
		v := make([]float64, embeddingDim)
		for j := 0; j < embeddingDim; j++ {
			v[j] = x.Value().At(i, j)
		}
		idx := lookupEmbedding(embedding, v)
		if idx == 1 {
			break
		}
		str += vocabs[idx]
	}
	return str
}
