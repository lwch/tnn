package sample

func onehot(x, size int) []float32 {
	ret := make([]float32, size)
	ret[x] = 1
	return ret
}

func zerohot(size int) []float32 {
	ret := make([]float32, size)
	return ret
}

func encode(vocabs []string, idx []int) string {
	var ret string
	for _, idx := range idx {
		if idx < 0 {
			ret += "<pad>"
			break
		}
		ret += vocabs[idx]
	}
	return ret
}

// Build 生成一个样本，输出: sequence, next word, padding mask
func Build(s *Sample, paddingSize int, embedding [][]float32, vocabs []string) ([]float32, []float32, []bool) {
	// fmt.Printf("%s => %s\n", encode(vocabs, x), encode(vocabs, []int{y}))
	embeddingSize := len(embedding[0])
	dx := make([]float32, 0, paddingSize*embeddingSize)
	paddingMask := make([]bool, 0, paddingSize)
	for _, v := range s.x {
		dx = append(dx, embedding[v]...)
		paddingMask = append(paddingMask, false)
	}
	for i := len(s.x); i < paddingSize; i++ {
		for j := 0; j < embeddingSize; j++ {
			dx = append(dx, 0)
		}
		paddingMask = append(paddingMask, true)
	}
	return dx, onehot(s.y, len(embedding)), paddingMask
}
