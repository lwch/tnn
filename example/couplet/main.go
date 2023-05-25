package main

import (
	"os"
)

func main() {
	if _, err := os.Stat(dataDir); os.IsNotExist(err) {
		download()
	}
	idx2vocab, vocab2idx := loadVocab()
	if _, err := os.Stat(modelDir); os.IsNotExist(err) {
		buildEmbedding(len(idx2vocab))
	}
	embedding := loadEmbedding(len(idx2vocab))
	trainX := loadData("train/in.txt", vocab2idx)
	trainY := loadData("train/out.txt", vocab2idx)
	train(trainX, trainY, embedding)
}
