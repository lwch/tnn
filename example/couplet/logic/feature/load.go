package feature

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/lwch/runtime"
)

func LoadVocab(dir string) ([]string, map[string]int) {
	f, err := os.Open(dir)
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

func LoadData(dir string, idx map[string]int, limit int) [][]int {
	f, err := os.Open(dir)
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
