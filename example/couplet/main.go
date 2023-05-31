package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

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

var cutCmd = cobra.Command{
	Use:  "cut [size]",
	Args: cobra.MinimumNArgs(1),
	Run:  runCut,
}

const trainXDir = "train/in.txt"
const trainYDir = "train/out.txt"

func main() {
	if _, err := os.Stat(dataDir); os.IsNotExist(err) {
		download()
	}

	rootCmd.AddCommand(&trainCmd)
	rootCmd.AddCommand(&predictCmd)
	rootCmd.AddCommand(&cutCmd)

	rootCmd.CompletionOptions.DisableDefaultCmd = true
	runtime.Assert(rootCmd.Execute())
}

func runTrain(*cobra.Command, []string) {
	idx2vocab, vocab2idx := loadVocab()
	if _, err := os.Stat(modelDir); os.IsNotExist(err) {
		buildEmbedding(len(idx2vocab))
	}
	embedding := loadEmbedding(len(idx2vocab))
	trainX := loadData(trainXDir, vocab2idx, -1)
	trainY := loadData(trainYDir, vocab2idx, -1)
	train(trainX, trainY, idx2vocab, embedding)
}

func runPredict(_ *cobra.Command, args []string) {
	idx2vocab, vocab2idx := loadVocab()
	embedding := loadEmbedding(len(idx2vocab))
	predict(args[0], idx2vocab, vocab2idx, embedding)
}

func runCut(_ *cobra.Command, args []string) {
	size, err := strconv.ParseInt(args[0], 10, 64)
	runtime.Assert(err)
	idx2Vocab, vocab2idx := loadVocab()
	xData := loadData(trainXDir, vocab2idx, int(size))
	yData := loadData(trainYDir, vocab2idx, int(size))
	build := func(data [][]int, dir string) {
		f, err := os.Create(filepath.Join(dataDir, dir))
		runtime.Assert(err)
		defer f.Close()
		for _, tks := range data {
			tokens := make([]string, 0, len(tks))
			for _, i := range tks {
				tk := idx2Vocab[i]
				if tk == "<s>" {
					continue
				}
				if tk == "</s>" {
					continue
				}
				tokens = append(tokens, tk)
			}
			fmt.Fprintln(f, strings.Join(tokens, " "))
		}
	}
	build(xData, trainXDir)
	build(yData, trainYDir)
	runtime.Assert(err)
}
