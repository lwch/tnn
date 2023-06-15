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

// Embedding 生成一个样本，返回内容：x, y, paddingIdx
func (s *Sample) Embedding(paddingSize int, embedding [][]float32) ([]float32, []int64, int) {
	embeddingSize := len(embedding[0])
	dx := make([]float32, 0, paddingSize*embeddingSize)
	dy := make([]int64, 0, paddingSize*len(embedding))
	for i := range s.x {
		dx = append(dx, embedding[s.x[i]]...)
		dy = append(dy, int64(s.y[i]))
	}
	for i := len(s.x); i < paddingSize; i++ {
		for j := 0; j < embeddingSize; j++ {
			dx = append(dx, 0)
		}
		dy = append(dy, 0)
	}
	return dx, dy, len(s.x)
}
