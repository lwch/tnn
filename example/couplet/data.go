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
)

const dataDir = "./data"
const downloadUrl = "https://github.com/wb14123/couplet-dataset/releases/latest/download/couplet.tar.gz"
const paddingSize = 74 // 最长为34*2

var paddingEmbedding []float64

func init() {
	for i := 0; i < embeddingDim; i++ {
		paddingEmbedding = append(paddingEmbedding, 0)
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

func loadData(dir string, idx map[string]int, limit int) [][]int {
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
		row = append(row, 1) // </s>
		data = append(data, row)
		if len(row) > max {
			max = len(row)
		}
		if limit > 0 && len(data) >= limit {
			break
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

func zerohot(size int) []float64 {
	ret := make([]float64, size)
	return ret
}

func encode(vocabs []string, idx []int) string {
	var ret string
	for _, idx := range idx {
		if idx == paddingIdx {
			break
		}
		ret += vocabs[idx]
	}
	return ret
}

const paddingIdx = 1000000

func build(x, y []int, z int, vocabs []string, embedding [][]float64) ([]float64, []float64, []bool) {
	// 输出: sentence, next word, padding mask
	// fmt.Println(encode(vocabs, x), "!!!", encode(vocabs, y), "!!!", encode(vocabs, []int{z}))
	dx := make([]float64, 0, unitSize)
	paddingMask := make([]bool, 0, paddingSize)
	for _, v := range x {
		dx = append(dx, embedding[v]...)
		paddingMask = append(paddingMask, false)
	}
	dx = append(dx, embedding[0]...) // <s>
	paddingMask = append(paddingMask, false)
	for _, v := range y {
		dx = append(dx, embedding[v]...)
		paddingMask = append(paddingMask, false)
	}
	dx = append(dx, embedding[1]...) // </s>
	paddingMask = append(paddingMask, false)
	for i := len(x) + len(y) + 2; i < paddingSize; i++ {
		dx = append(dx, paddingEmbedding...)
		paddingMask = append(paddingMask, true)
	}
	dz := make([]float64, 0, len(embedding))
	if z == paddingIdx {
		dz = append(dz, zerohot(len(embedding))...)
	} else {
		dz = append(dz, onehot(z, len(embedding))...)
	}
	return dx, dz, paddingMask
}
