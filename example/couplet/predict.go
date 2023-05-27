package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/model"
	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
)

func predict(str string, vocabs []string, vocab2idx map[string]int, embedding [][]float64) {
	dir := filepath.Join(modelDir, "encoder.model")
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		panic("encoder model not found")
	}
	var enc model.Model
	runtime.Assert(enc.Load(dir))
	dir = filepath.Join(modelDir, "decoder.model")
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		panic("decoder model not found")
	}
	var dec model.Model
	runtime.Assert(dec.Load(dir))
	encoder = enc.Layers()
	decoder = dec.Layers()
	dx := make([]int, 0, len(str)*embeddingDim)
	var size int
	for _, ch := range str {
		dx = append(dx, vocab2idx[string(ch)])
		size++
	}
	dy := make([]int, 0, len(str)*embeddingDim)
	output := make([]float64, 0, len(str)*embeddingDim)
	for i := 0; i < size; i++ {
		x, y, _ := buildTensor([][]int{dx}, [][]int{dy}, embedding, false)
		pred := forward(x, y, false)
		predEmbedding := pred.Value().RowView(0).(*mat.VecDense).RawVector().Data
		output = append(output, predEmbedding...)
		label := lookupEmbedding(embedding, predEmbedding)
		dy = append(dy, label)
	}
	fmt.Println(tensorStr(tensor.New(output, 1, size*embeddingDim), vocabs, embedding))
	// var pred string
	// mix := embedding[int(time.Now().UnixNano())%len(embedding)]
	// y := tensor.New(mix, 1, embeddingDim)
	// for _, ch := range str {
	// 	x := tensor.New(embedding[vocab2idx[string(ch)]], 1, embeddingDim)
	// 	for _, layer := range encoder.Layers() {
	// 		x = layer.Forward(x, false)
	// 	}
	// 	layers := decoder.Layers()
	// 	y = layers[0].Forward(y, false)
	// 	y = layers[1].Forward(y, false)
	// 	y = layers[2].(*layer.SelfAttention).ForwardQKV(y, x, y, false)
	// 	for i := 3; i < len(layers); i++ {
	// 		y = layers[i].Forward(y, false)
	// 	}
	// 	pred += tensorStr(y, vocabs, embedding)[0]
	// }
	// fmt.Println(pred)
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
	rows, cols := x.Dims()
	var ret []string
	for row := 0; row < rows; row++ {
		var str string
		for start := 0; start < cols/embeddingDim; start++ {
			v := make([]float64, embeddingDim)
			for j := 0; j < embeddingDim; j++ {
				v[j] = x.Value().At(row, start*embeddingDim+j)
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
