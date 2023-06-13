package feature

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/lwch/runtime"
)

// Build 生成训练样本文件
func Build(data [][]int, vocabs []string, dir string) map[string]int {
	runtime.Assert(os.MkdirAll(filepath.Dir(dir), 0755))
	f, err := os.Create(dir)
	runtime.Assert(err)
	defer f.Close()
	ret := make(map[string]int) // token => 字频
	for _, tks := range data {
		tokens := make([]string, 0, len(tks))
		for _, i := range tks {
			tk := vocabs[i]
			if tk == "<s>" {
				continue
			}
			if tk == "</s>" {
				continue
			}
			tokens = append(tokens, tk)
			ret[tk]++
		}
		fmt.Fprintln(f, strings.Join(tokens, " "))
	}
	return ret
}
