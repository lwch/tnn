package main

import (
	"archive/tar"
	"bufio"
	"compress/gzip"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/tensor"
)

const dataDir = "./data"
const downloadUrl = "https://github.com/wb14123/couplet-dataset/releases/latest/download/couplet.tar.gz"
const paddingSize = 40

var paddingEmbedding []float64

func init() {
	for i := 0; i < embeddingDim; i++ {
		paddingEmbedding = append(paddingEmbedding, -1e9)
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
			if len(v) == 0 {
				continue
			}
			row = append(row, idx[v])
		}
		// row = append(row, 1) // </s>
		data = append(data, row)
		if len(row) > max {
			max = len(row)
		}
	}
	fmt.Printf("max token size in %s: %d\n", dir, max)
	return data
}

func onehot(x, size int) []float64 {
	ret := make([]float64, size)
	ret[x] = 1
	return ret
}

func encode(vocabs []string, idx []int) string {
	var ret string
	for _, idx := range idx {
		ret += vocabs[idx]
	}
	return ret
}

func buildTensor(x, y [][]int, vocabs []string, embedding [][]float64, training bool) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	dx := make([]float64, 0, len(x)*unitSize)
	dy := make([]float64, 0, len(y)*unitSize)
	dz := make([]float64, 0, len(y)*len(embedding))
	rows := 0
	add := func(x, y, yy []int, z int) {
		// fmt.Println(encode(vocabs, x), "!!!", encode(vocabs, y), "!!!", encode(vocabs, []int{z}))
		for _, v := range x {
			dx = append(dx, embedding[v]...)
		}
		dy = append(dy, embedding[0]...) // <s>
		for _, v := range y {
			dy = append(dy, embedding[v]...)
		}
		dz = append(dz, onehot(z, len(embedding))...)
		for i := len(x); i < paddingSize; i++ {
			dx = append(dx, paddingEmbedding...)
		}
		for i := len(y) + 1; i < paddingSize; i++ {
			dy = append(dy, paddingEmbedding...)
		}
		rows++
	}
	if training {
		for i := range y {
			for j := 0; j < len(y[i]); j++ {
				// if y[i][j] == 1 { // </s>
				// 	break
				// }
				add(x[i], y[i][:j], y[i], y[i][j])
			}
			// add(x[i], y[i])
		}
		dxa := make([]float64, unitSize)
		dya := make([]float64, unitSize)
		dza := make([]float64, len(embedding))
		rand.Shuffle(rows, func(i, j int) {
			copy(dxa, dx[i*unitSize:(i+1)*unitSize])
			copy(dya, dy[i*unitSize:(i+1)*unitSize])
			copy(dza, dz[i*len(embedding):(i+1)*len(embedding)])
			copy(dx[i*unitSize:(i+1)*unitSize], dx[j*unitSize:(j+1)*unitSize])
			copy(dy[i*unitSize:(i+1)*unitSize], dy[j*unitSize:(j+1)*unitSize])
			copy(dz[i*len(embedding):(i+1)*len(embedding)], dz[j*len(embedding):(j+1)*len(embedding)])
			copy(dx[j*unitSize:(j+1)*unitSize], dxa)
			copy(dy[j*unitSize:(j+1)*unitSize], dya)
			copy(dz[j*len(embedding):(j+1)*len(embedding)], dza)
		})
	} else {
		add(x[0], y[0], y[0], 1)
	}
	return tensor.New(dx, rows, unitSize),
		tensor.New(dy, rows, unitSize),
		tensor.New(dz, rows, len(embedding))
}
