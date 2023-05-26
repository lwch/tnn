package main

import (
	"os"

	"github.com/lwch/runtime"
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
	Use:  "predict [content]",
	Args: cobra.MinimumNArgs(1),
	Run:  runPredict,
}

func main() {
	if _, err := os.Stat(dataDir); os.IsNotExist(err) {
		download()
	}

	rootCmd.AddCommand(&trainCmd)
	rootCmd.AddCommand(&predictCmd)

	rootCmd.CompletionOptions.DisableDefaultCmd = true
	runtime.Assert(rootCmd.Execute())
}

func runTrain(*cobra.Command, []string) {
	idx2vocab, vocab2idx := loadVocab()
	if _, err := os.Stat(modelDir); os.IsNotExist(err) {
		buildEmbedding(len(idx2vocab))
	}
	embedding := loadEmbedding(len(idx2vocab))
	trainX := loadData("train/in2.txt", vocab2idx)
	trainY := loadData("train/out2.txt", vocab2idx)
	train(trainX, trainY, embedding)
}

func runPredict(_ *cobra.Command, args []string) {
	idx2vocab, vocab2idx := loadVocab()
	embedding := loadEmbedding(len(idx2vocab))
	predict(args[0], idx2vocab, vocab2idx, embedding)
}
