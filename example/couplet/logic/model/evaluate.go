package model

import (
	"fmt"
	"sort"
)

// Evaluate 根据输入内容进行推理
func (m *Model) Evaluate(str string) string {
	// dx := make([]int, 0, len(str))
	// var size int
	// for _, ch := range str {
	// 	dx = append(dx, m.vocabsIdx[string(ch)])
	// 	size++
	// }
	// dx = append(dx, 1) // </s>
	// dy := make([]int, 0, len(str))
	// dy = append(dy, 0) // <s>
	// for i := 0; i < size; i++ {
	// 	x, _, paddingMask := sample.Build(append(dx, dy...), 0, paddingSize, m.embedding, m.vocabs)
	// 	pred := m.forward(tensor.New(x, 1, unitSize), buildPaddingMasks([][]bool{paddingMask}), false)
	// 	predProb := pred.Value().RowView(0).(*mat32.VecDense).RawVector().Data
	// 	label := lookup(predProb, m.vocabs)
	// 	dy = append(dy, label)
	// }
	// return values(m.vocabs, dy[1:])

	// TODO
	return ""
}

func lookup(prob []float32, vocabs []string) int {
	var max float32
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
