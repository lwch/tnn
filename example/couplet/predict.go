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
	dir := filepath.Join(modelDir, "couplet.model")
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		panic("model not found")
	}
	var m model.Model
	runtime.Assert(m.Load(dir))
	layers = m.Layers()
	dx := make([]int, 0, len(str))
	var size int
	for _, ch := range str {
		dx = append(dx, vocab2idx[string(ch)])
		size++
	}
	dy := make([]int, 0, len(str))
	// n := rand.Intn(len(embedding))
	// if n < 2 {
	// 	n += 2
	// }
	// dy = append(dy, n)
	for i := 0; i < size; i++ {
		x, y, _ := buildTensor([][]int{dx}, [][]int{dy}, vocabs, embedding, false)
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
	for i := 0; i < len(prob); i++ {
		if prob[i] > max {
			max = prob[i]
			idx = i
		}
	}
	// kv := make(map[float64][]int)
	// for i := 0; i < len(prob); i++ {
	// 	kv[prob[i]] = append(kv[prob[i]], i)
	// }
	// sort.Float64s(prob)
	// fmt.Println(kv[prob[len(prob)-4]])
	return idx
}

func values(vocabs []string, idx []int) string {
	var str string
	for _, i := range idx {
		str += vocabs[i]
	}
	return str
}
