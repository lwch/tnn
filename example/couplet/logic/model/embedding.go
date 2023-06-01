package model

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/initializer"
)

// buildEmbedding 生成每个词的embedding
func (m *Model) buildEmbedding(dir string) {
	fmt.Println("build embedding...")
	init := initializer.NewXavierUniform(1)
	data := init.RandShape(len(m.vocabs), embeddingDim)
	runtime.Assert(os.MkdirAll(filepath.Dir(dir), 0755))
	f, err := os.Create(dir)
	runtime.Assert(err)
	defer f.Close()
	runtime.Assert(binary.Write(f, binary.BigEndian, data))
}

// loadEmbedding 加载每个词的embedding
func (m *Model) loadEmbedding(dir string) [][]float32 {
	fmt.Println("load embedding...")
	f, err := os.Open(dir)
	runtime.Assert(err)
	defer f.Close()
	var ret [][]float32
	for i := 0; i < len(m.vocabs); i++ {
		data := make([]float32, embeddingDim)
		runtime.Assert(binary.Read(f, binary.BigEndian, &data))
		ret = append(ret, data)
	}
	return ret
}
