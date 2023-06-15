package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/example/couplet/logic/feature"
	"github.com/lwch/tnn/example/couplet/logic/model"
	"github.com/spf13/cobra"
)

var rootCmd = cobra.Command{
	Use: "couplet",
}

var downloadCmd = cobra.Command{
	Use: "download",
	Run: runDownload,
}

var trainCmd = cobra.Command{
	Use: "train",
	Run: runTrain,
}

var evaluateCmd = cobra.Command{
	Use:  "evaluate [content]",
	Args: cobra.MinimumNArgs(1),
	Run:  runEvaluate,
}

var cutCmd = cobra.Command{
	Use:  "cut [size]",
	Args: cobra.MinimumNArgs(1),
	Run:  runCut,
}

const dataDir = "./data"
const sampleDir = "./sample"
const modelDir = "./model"

var evaluateModelDir string

func main() {
	evaluateCmd.Flags().StringVar(&evaluateModelDir, "model", "", "model directory")

	rootCmd.AddCommand(&downloadCmd)
	rootCmd.AddCommand(&trainCmd)
	rootCmd.AddCommand(&evaluateCmd)
	rootCmd.AddCommand(&cutCmd)

	rootCmd.CompletionOptions.DisableDefaultCmd = true
	runtime.Assert(rootCmd.Execute())
}

func runDownload(*cobra.Command, []string) {
	feature.Download(dataDir)
}

func runCut(_ *cobra.Command, args []string) {
	size, err := strconv.ParseInt(args[0], 10, 64)
	runtime.Assert(err)
	vocabs, vocab2idx := feature.LoadVocab(filepath.Join(dataDir, "vocabs"))
	xData := feature.LoadData(filepath.Join(dataDir, "train", "in.txt"), vocab2idx, int(size))
	yData := feature.LoadData(filepath.Join(dataDir, "train", "out.txt"), vocab2idx, int(size))
	xtokens := feature.Build(xData, vocabs, filepath.Join(sampleDir, "in.txt"))
	ytokens := feature.Build(yData, vocabs, filepath.Join(sampleDir, "out.txt"))
	merge := make(map[string]int)
	for tk, cnt := range xtokens {
		merge[tk] += cnt
	}
	for tk, cnt := range ytokens {
		merge[tk] += cnt
	}
	vocabs = vocabs[:0]
	for tk := range merge {
		vocabs = append(vocabs, tk)
	}
	sort.Slice(vocabs, func(i, j int) bool {
		return merge[vocabs[i]] > merge[vocabs[j]]
	})
	vocabs = append([]string{"<s>", "</s>"}, vocabs...)
	err = os.WriteFile(filepath.Join(sampleDir, "vocabs"), []byte(strings.Join(vocabs, "\n")), 0644)
	runtime.Assert(err)
}

func runTrain(*cobra.Command, []string) {
	m := model.New()
	m.Train(sampleDir, modelDir)
}

func runEvaluate(_ *cobra.Command, args []string) {
	var m model.Model
	m.Load(evaluateModelDir)
	fmt.Println(m.Evaluate(args[0]))
}
