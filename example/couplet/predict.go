package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/model"
	"github.com/lwch/tnn/nn/tensor"
)

func predict(str string, vocabs []string, vocab2idx map[string]int, embedding [][]float64) {
	dir := filepath.Join(modelDir, "encoder.model")
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		panic("encoder model not found")
	}
	x := make([]float64, 0, len(str)*embeddingDim)
	var size int
	for _, ch := range str {
		x = append(x, embedding[vocab2idx[string(ch)]]...)
		size++
	}
	x = append(x, embedding[1]...) // </s>
	size++
	var m model.Model
	runtime.Assert(m.Load(dir))
	y := m.Predict(tensor.New(x, size, embeddingDim))
	fmt.Println(tensorStr(y, vocabs, embedding))
}

func lookupEmbedding(embedding [][]float64, v []float64) int {
	fmt.Println(v)
	min := math.MaxFloat64
	ret := 0
	for i := 0; i < len(embedding); i++ {
		d := 0.0
		for j := 0; j < len(embedding[i]); j++ {
			d += math.Pow(embedding[i][j]-v[j], 2)
		}
		if d < min {
			min = d
			ret = i
		}
	}
	return ret
}

func tensorStr(x *tensor.Tensor, vocabs []string, embedding [][]float64) string {
	rows, _ := x.Dims()
	var str string
	for i := 0; i < rows; i++ {
		v := make([]float64, embeddingDim)
		for j := 0; j < embeddingDim; j++ {
			v[j] = x.Value().At(i, j)
		}
		idx := lookupEmbedding(embedding, v)
		if idx == 1 {
			break
		}
		str += vocabs[idx]
	}
	return str
}
