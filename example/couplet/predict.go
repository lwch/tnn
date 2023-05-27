package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/model"
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
	dx := make([]int, 0, len(str))
	var size int
	for _, ch := range str {
		dx = append(dx, vocab2idx[string(ch)])
		size++
	}
	dy := make([]int, 0, len(str))
	for i := 0; i < size; i++ {
		x, y, _ := buildTensor([][]int{dx}, [][]int{dy}, embedding, false)
		pred := forward(x, y, false)
		predProb := pred.Value().RowView(0).(*mat.VecDense).RawVector().Data
		label := lookup(predProb)
		dy = append(dy, label)
	}
	fmt.Println(values(vocabs, dy))
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

func lookup(prob []float64) int {
	var max float64
	var idx int
	for i, v := range prob {
		if v > max {
			max = v
			idx = i
		}
	}
	return idx
}

func values(vocabs []string, idx []int) string {
	var str string
	for _, i := range idx {
		str += vocabs[i]
	}
	return str
}
