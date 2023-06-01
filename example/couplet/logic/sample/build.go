package sample

func onehot(x, size int) []float64 {
	ret := make([]float64, size)
	ret[x] = 1
	return ret
}

func zerohot(size int) []float64 {
	ret := make([]float64, size)
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
func Build(x []int, y, paddingSize int, embedding [][]float64, vocabs []string) ([]float64, []float64, []bool) {
	// fmt.Printf("%s => %s\n", encode(vocabs, x), encode(vocabs, []int{y}))
	embeddingSize := len(embedding[0])
	dx := make([]float64, 0, paddingSize*embeddingSize)
	paddingMask := make([]bool, 0, paddingSize)
	for _, v := range x {
		dx = append(dx, embedding[v]...)
		paddingMask = append(paddingMask, false)
	}
	for i := len(x); i < paddingSize; i++ {
		for j := 0; j < embeddingSize; j++ {
			dx = append(dx, 0)
		}
		paddingMask = append(paddingMask, true)
	}
	dz := make([]float64, 0, len(embedding))
	if y < 0 {
		dz = append(dz, zerohot(len(embedding))...)
	} else {
		dz = append(dz, onehot(y, len(embedding))...)
	}
	return dx, dz, paddingMask
}
