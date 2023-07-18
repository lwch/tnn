package model

import (
	"fmt"
	"math"
	"sort"

	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/example/couplet/logic/sample"
)

// Evaluate 根据输入内容进行推理
func (m *Model) Evaluate(str string) string {
	defer storage.GC()
	dx := make([]int, 0, len(str))
	var size int
	for _, ch := range str {
		dx = append(dx, m.vocabsIdx[string(ch)])
		size++
	}
	fmt.Printf("inputs: %v\n", dx)
	dy := make([]int, len(dx))
	x, _, p := sample.New(dx, dy).Embedding(paddingSize, m.embedding)
	pred := m.forward(
		tensor.FromFloat32(storage, x, tensor.WithShapes(1, paddingSize, embeddingDim)),
		[]int{p}, false)
	predProbs := pred.Float32Value()
	dy = dy[:0]
	for i := 0; i < size; i++ {
		start := i * len(m.vocabs)
		label := lookup(predProbs[start:start+len(m.vocabs)], m.vocabs)
		dy = append(dy, label)
	}
	return values(m.vocabs, dy)
}

func lookup(prob []float32, vocabs []string) int {
	max := float32(-math.MaxFloat32)
	var idx int
	for i := 0; i < len(prob); i++ {
		if prob[i] > max {
			max = prob[i]
			idx = i
		}
	}
	kv := make(map[float32][]string)
	for i := 0; i < len(prob); i++ {
		kv[prob[i]] = append(kv[prob[i]], vocabs[i])
	}
	sort.Slice(prob, func(i, j int) bool {
		return prob[i] < prob[j]
	})
	left := make(map[float32][]string)
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
