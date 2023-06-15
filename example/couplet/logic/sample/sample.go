package sample

type Sample struct {
	x []int
	y []int
}

func New(x, y []int) *Sample {
	return &Sample{x: x, y: y}
}

// Encode 将内容转换为文字
func (s *Sample) Encode(vocabs []string) (string, string) {
	var x string
	for _, idx := range s.x {
		x += vocabs[idx]
	}
	var y string
	for _, idx := range s.y {
		y += vocabs[idx]
	}
	return x, y
}

func onehot(x, size int) []float32 {
	ret := make([]float32, size)
	ret[x] = 1
	return ret
}

// Embedding 生成一个样本
func (s *Sample) Embedding(paddingSize int, embedding [][]float32) ([]float32, []float32) {
	embeddingSize := len(embedding[0])
	dx := make([]float32, 0, paddingSize*embeddingSize)
	dy := make([]float32, 0, paddingSize*len(embedding))
	for i := range s.x {
		dx = append(dx, embedding[s.x[i]]...)
		dy = append(dy, onehot(s.y[i], len(embedding))...)
	}
	for i := len(s.x); i < paddingSize; i++ {
		for j := 0; j < embeddingSize; j++ {
			dx = append(dx, 0)
		}
		for j := 0; j < len(embedding); j++ {
			dy = append(dy, 0)
		}
	}
	return dx, dy
}
