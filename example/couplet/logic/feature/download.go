package feature

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/lwch/runtime"
)

const downloadUrl = "https://github.com/wb14123/couplet-dataset/releases/latest/download/couplet.tar.gz"

func Download(dir string) {
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
		dir := filepath.Join(dir, strings.TrimPrefix(hdr.Name, "couplet"))
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
