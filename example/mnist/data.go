package main

import (
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/lwch/runtime"
)

var trainData = "https://gist.github.com/lwch/e4b01d01ee3f2549fa32379fe2c6d79b/raw/0c4a44f202af9772b4ff61371a0145f3010bb741/train-images-idx3-ubyte"
var trainLabel = "https://gist.github.com/lwch/e4b01d01ee3f2549fa32379fe2c6d79b/raw/0c4a44f202af9772b4ff61371a0145f3010bb741/train-labels-idx1-ubyte"
var testData = "https://gist.github.com/lwch/e4b01d01ee3f2549fa32379fe2c6d79b/raw/0c4a44f202af9772b4ff61371a0145f3010bb741/t10k-images-idx3-ubyte"
var testLabel = "https://gist.github.com/lwch/e4b01d01ee3f2549fa32379fe2c6d79b/raw/0c4a44f202af9772b4ff61371a0145f3010bb741/t10k-labels-idx1-ubyte"

func download() {
	downloadData(trainData, filepath.Join(dataDir, "train"))
	downloadLabel(trainLabel, filepath.Join(dataDir, "train"))
	downloadData(testData, filepath.Join(dataDir, "test"))
	downloadLabel(testLabel, filepath.Join(dataDir, "test"))
}

func downloadData(url, dir string) {
	fmt.Printf("download %s...\n", url)
	url = "https://ghproxy.com/" + url
	resp, err := http.Get(url)
	runtime.Assert(err)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		panic("download data failed")
	}
	var hdr struct {
		Magic uint32
		Count uint32
		Rows  uint32
		Cols  uint32
	}
	err = binary.Read(resp.Body, binary.BigEndian, &hdr)
	runtime.Assert(err)
	if hdr.Magic != 2051 {
		panic("invalid magic")
	}
	save := func(dir string, img image.Image) {
		f, err := os.Create(dir)
		runtime.Assert(err)
		defer f.Close()
		err = png.Encode(f, img)
		runtime.Assert(err)
	}
	for i := 0; i < int(hdr.Count); i++ {
		img := image.NewGray(image.Rect(0, 0, int(hdr.Cols), int(hdr.Rows)))
		for y := 0; y < int(hdr.Rows); y++ {
			for x := 0; x < int(hdr.Cols); x++ {
				var v uint8
				err = binary.Read(resp.Body, binary.BigEndian, &v)
				runtime.Assert(err)
				img.SetGray(x, y, color.Gray{v})
			}
		}
		os.MkdirAll(dir, 0755)
		save(filepath.Join(dir, fmt.Sprintf("%d.png", i)), img)
		if i != 0 && i%1000 == 0 {
			fmt.Printf("download %d/%d...\n", i, hdr.Count)
		}
	}
}

func downloadLabel(url, dir string) {
	fmt.Printf("download %s...\n", url)
	url = "https://ghproxy.com/" + url
	resp, err := http.Get(url)
	runtime.Assert(err)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		panic("download label failed")
	}
	var hdr struct {
		Magic uint32
		Count uint32
	}
	err = binary.Read(resp.Body, binary.BigEndian, &hdr)
	runtime.Assert(err)
	if hdr.Magic != 2049 {
		panic("invalid magic")
	}
	os.MkdirAll(dir, 0755)
	f, err := os.Create(filepath.Join(dir, "label"))
	runtime.Assert(err)
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	runtime.Assert(err)
}

type dataSet struct {
	rows, cols int
	images     []image.Image
	labels     []uint8
}

func loadData(dir string) dataSet {
	files, err := filepath.Glob(filepath.Join(dir, "*.png"))
	runtime.Assert(err)
	sort.Slice(files, func(i, j int) bool {
		aFile := filepath.Base(files[i])
		bFile := filepath.Base(files[j])
		aFile = strings.TrimSuffix(aFile, filepath.Ext(aFile))
		bFile = strings.TrimSuffix(bFile, filepath.Ext(bFile))
		a, _ := strconv.ParseInt(aFile, 10, 64)
		b, _ := strconv.ParseInt(bFile, 10, 64)
		return a < b
	})
	label, err := os.Open(filepath.Join(dir, "label"))
	runtime.Assert(err)
	defer label.Close()
	load := func(file string) image.Image {
		f, err := os.Open(file)
		runtime.Assert(err)
		defer f.Close()
		img, err := png.Decode(f)
		runtime.Assert(err)
		return img
	}
	var ret dataSet
	for _, file := range files {
		img := load(file)
		pt := img.Bounds().Max
		ret.rows = pt.X
		ret.cols = pt.Y
		ret.images = append(ret.images, img)
		var v uint8
		err = binary.Read(label, binary.BigEndian, &v)
		runtime.Assert(err)
		ret.labels = append(ret.labels, v)
	}
	return ret
}