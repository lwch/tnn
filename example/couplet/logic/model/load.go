package model

import (
	"os"
	"path/filepath"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/example/couplet/logic/feature"
	"github.com/lwch/tnn/nn/model"
)

// Load 加载模型
func (m *Model) Load(dir string) {
	if _, err := os.Stat(filepath.Join(dir, "couplet.model")); os.IsNotExist(err) {
		panic("model not found")
	}
	var md model.Model
	runtime.Assert(md.Load(filepath.Join(dir, "couplet.model")))
	m.layers = md.Layers()

	if _, err := os.Stat(filepath.Join(dir, "vocabs")); os.IsNotExist(err) {
		panic("vocabs not found")
	}
	m.vocabs, m.vocabsIdx = feature.LoadVocab(filepath.Join(dir, "vocabs"))

	if _, err := os.Stat(filepath.Join(dir, "embedding")); os.IsNotExist(err) {
		panic("embedding not found")
	}
	m.embedding = m.loadEmbedding(filepath.Join(dir, "embedding"))
}
