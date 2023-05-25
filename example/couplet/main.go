package main

import (
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = cobra.Command{
	Use: "couplet",
}

var trainCmd = cobra.Command{
	Use: "train",
	Run: runTrain,
}

var predictCmd = cobra.Command{
	Use: "predict",
	Run: runPredict,
}

func main() {
	if _, err := os.Stat(dataDir); os.IsNotExist(err) {
		download()
	}

	rootCmd.AddCommand(&trainCmd)
	rootCmd.AddCommand(&predictCmd)
}

func runTrain(*cobra.Command, []string) {
	idx2vocab, vocab2idx := loadVocab()
	if _, err := os.Stat(modelDir); os.IsNotExist(err) {
		buildEmbedding(len(idx2vocab))
	}
	embedding := loadEmbedding(len(idx2vocab))
	trainX := loadData("train/in.txt", vocab2idx)
	trainY := loadData("train/out.txt", vocab2idx)
	train(trainX, trainY, embedding)
}

func runPredict(*cobra.Command, []string) {
}
