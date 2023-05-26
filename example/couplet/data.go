package main

import (
	"archive/tar"
	"bufio"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/tensor"
)

const dataDir = "./data"
const downloadUrl = "https://github.com/wb14123/couplet-dataset/releases/latest/download/couplet.tar.gz"
const padSize = 40

var padEmbedding []float64

func init() {
	padEmbedding = make([]float64, embeddingDim)
	for i := range padEmbedding {
		padEmbedding[i] = 1
	}
}

func download() {
	fmt.Println("download dataset...")
	u := downloadUrl
	u = "https://ghproxy.com/" + u
	req, err := http.NewRequest(http.MethodGet, u, nil)
	runtime.Assert(err)
	rep, err := http.DefaultClient.Do(req)
	runtime.Assert(err)
	defer rep.Body.Close()
	gr, err := gzip.NewReader(rep.Body)
	runtime.Assert(err)
	defer gr.Close()
	tr := tar.NewReader(gr)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return
		}
		runtime.Assert(err)
		if strings.HasSuffix(hdr.Name, ".swp") {
			continue
		}
		dir := filepath.Join(dataDir, strings.TrimPrefix(hdr.Name, "couplet"))
		if hdr.FileInfo().IsDir() {
			os.MkdirAll(dir, 0755)
			continue
		}
		fmt.Printf("saving %s\n", hdr.Name)
		f, err := os.Create(dir)
		runtime.Assert(err)
		defer f.Close()
		_, err = io.Copy(f, tr)
		runtime.Assert(err)
	}
}

func loadVocab() ([]string, map[string]int) {
	f, err := os.Open(filepath.Join(dataDir, "vocabs"))
	runtime.Assert(err)
	defer f.Close()
	s := bufio.NewScanner(f)
	var idx2vocab []string
	vocab2idx := make(map[string]int)
	for s.Scan() {
		str := strings.TrimSpace(s.Text())
		if len(str) == 0 {
			continue
		}
		vocab2idx[str] = len(idx2vocab)
		idx2vocab = append(idx2vocab, str)
	}
	return idx2vocab, vocab2idx
}

func loadData(dir string, idx map[string]int) [][]int {
	f, err := os.Open(filepath.Join(dataDir, dir))
	runtime.Assert(err)
	defer f.Close()
	s := bufio.NewScanner(f)
	var data [][]int
	max := 0
	for s.Scan() {
		var row []int
		vocab := strings.Split(s.Text(), " ")
		for _, v := range vocab {
			row = append(row, idx[v])
		}
		row = append(row, 1) // </s>
		data = append(data, row)
		if len(row) > max {
			max = len(row)
		}
	}
	fmt.Printf("max token size in %s: %d\n", dir, max)
	return data
}

func buildTensor(x, y [][]int, embedding [][]float64) (*tensor.Tensor, *tensor.Tensor) {
	dx := make([]float64, 0, len(x)*embeddingDim)
	dy := make([]float64, 0, len(y)*embeddingDim)
	rows := 0
	add := func(x, y []int) {
		for _, ch := range x {
			dx = append(dx, embedding[ch]...)
		}
		for i := len(x); i < padSize; i++ {
			dx = append(dx, padEmbedding...)
		}
		for _, ch := range y {
			dy = append(dy, embedding[ch]...)
		}
		for i := len(y); i < padSize; i++ {
			dy = append(dy, padEmbedding...)
		}
		// dx = append(dx, embedding[x]...)
		// dy = append(dy, embedding[y]...)
		rows++
	}
	for i := range x {
		// for j := range x[i] {
		// 	if x[i][j] == 1 { // </s>
		// 		add(x[i][j], y[i][j])
		// 		break
		// 	}
		// 	add(x[i][j], y[i][j])
		// }
		add(x[i], y[i])
	}
	return tensor.New(dx, rows, unitSize),
		tensor.New(dy, rows, unitSize)
}
