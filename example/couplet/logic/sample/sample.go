package sample

type Sample struct {
	x []int
	y int
}

func New(x []int, y int) *Sample {
	return &Sample{x: x, y: y}
}

// Encode 将内容转换为文字
func (s *Sample) Encode(vocabs []string) (string, string) {
	var x string
	for _, idx := range s.x {
		x += vocabs[idx]
	}
	return x, vocabs[s.y]
}

func onehot(x, size int) []float32 {
	ret := make([]float32, size)
	ret[x] = 1
	return ret
}

// Embedding 生成一个样本，输出: sequence, next word
func (s *Sample) Embedding(paddingSize int, embedding [][]float32) ([]float32, []float32) {
	embeddingSize := len(embedding[0])
	dx := make([]float32, 0, paddingSize*embeddingSize)
	for _, v := range s.x {
		dx = append(dx, embedding[v]...)
	}
	for i := len(s.x); i < paddingSize; i++ {
		for j := 0; j < embeddingSize; j++ {
			dx = append(dx, 0)
		}
	}
	return dx, onehot(s.y, len(embedding))
}
