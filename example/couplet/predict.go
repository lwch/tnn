package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/model"
	"github.com/lwch/tnn/nn/tensor"
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
	for i := 0; i < size; i++ {
		// x, y, _ := buildTensor([][]int{dx}, [][]int{dy}, vocabs, embedding, false)
		fmt.Println(dx, dy)
		x, _, paddingMask := build(dx, dy, 0, vocabs, embedding)
		pred := forward(tensor.New(x, 1, unitSize), buildPaddingMasks([][]bool{paddingMask}), false)
		predProb := pred.Value().RowView(0).(*mat.VecDense).RawVector().Data
		label := lookup(predProb)
		dy = append(dy, label)
	}
	fmt.Println(values(vocabs, dy))
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
	kv := make(map[float64][]int)
	for i := 0; i < len(prob); i++ {
		kv[prob[i]] = append(kv[prob[i]], i)
	}
	sort.Float64s(prob)
	left := make(map[float64][]int)
	for i := 0; i < 3; i++ {
		idx := len(prob) - i - 1
		left[prob[idx]] = kv[prob[idx]]
	}
	fmt.Println(left)
	return idx
}

func values(vocabs []string, idx []int) string {
	var str string
	for _, i := range idx {
		str += vocabs[i]
	}
	return str
}
