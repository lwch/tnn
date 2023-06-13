package model

import (
	"fmt"
	"io"
	_ "net/http/pprof"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"

	"github.com/lwch/gotorch/loss"
	"github.com/lwch/gotorch/mmgr"
	"github.com/lwch/gotorch/optimizer"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/net"
)

const (
	statusTrain = iota
	statusEvaluate
)

var lossFunc = loss.NewMse
var storage = mmgr.New()

// Model 模型
type Model struct {
	// 模型定义
	attn    []*transformer
	flatten *layer.Flatten
	relu    *activation.ReLU
	output  *layer.Dense

	// 运行时
	epoch    int           // 当前训练到第几个迭代
	current  atomic.Uint64 // 当前迭代已训练多少个样本
	total    int           // 样本总数
	status   int           // 当前运行状态
	chUpdate chan struct{} // 梯度更新信号
	modelDir string        // 模型保存路径

	vocabs    []string
	vocabsIdx map[string]int
	trainX    [][]int
	trainY    [][]int
	embedding [][]float32
	loss      loss.Loss
	optimizer optimizer.Optimizer
}

// New 创建空模型
func New() *Model {
	return &Model{
		chUpdate: make(chan struct{}),
	}
}

// build 生成模型
func (m *Model) build() {
	for i := 0; i < transformerSize; i++ {
		m.attn = append(m.attn, newTransformer())
	}
	m.flatten = layer.NewFlatten()
	m.relu = activation.NewReLU()
	m.output = layer.NewDense(len(m.vocabs))
}

func (m *Model) params() []*tensor.Tensor {
	var ret []*tensor.Tensor
	for _, attn := range m.attn {
		ret = append(ret, attn.params()...)
	}
	for _, p := range m.output.Params() {
		ret = append(ret, p)
	}
	return ret
}

// showProgress 显示进度
func (m *Model) showProgress() {
	tk := time.NewTicker(time.Second)
	defer tk.Stop()
	for {
		<-tk.C
		status := "train"
		if m.status == statusEvaluate {
			status = "evaluate"
		}
		fmt.Printf("%s: %d/%d\r", status, m.current.Load(), m.total)
	}
}

// save 保存模型
func (m *Model) save() {
	var net net.Net
	for _, attn := range m.attn {
		net.Add(attn.layers()...)
	}
	net.Add(m.flatten, m.relu, m.output)
	err := net.Save(filepath.Join(m.modelDir, "couplet.model"))
	runtime.Assert(err)
	fmt.Println("model saved")
}

// copyVocabs 拷贝vocabs文件到model下
func (m *Model) copyVocabs(dir string) {
	src, err := os.Open(dir)
	runtime.Assert(err)
	defer src.Close()
	dst, err := os.Create(filepath.Join(m.modelDir, "vocabs"))
	runtime.Assert(err)
	defer dst.Close()
	_, err = io.Copy(dst, src)
	runtime.Assert(err)
}

// forward 正向迭代
func (m *Model) forward(x *tensor.Tensor, paddingMasks []*tensor.Tensor, train bool) *tensor.Tensor {
	// x = x.Add(buildPositionEmbedding(batchSize)) // 添加位置信息
	y := x
	for _, attn := range m.attn {
		y = attn.forward(y)
	}
	y = m.flatten.Forward(y) // flatten
	y = m.relu.Forward(y)    // relu
	y = m.output.Forward(y)  // output
	y = y.Softmax(-1)        // softmax
	return y
}

func (m *Model) loadFrom(net *net.Net) {
	layers := net.Layers()
	idx := 0
	for i := 0; i < transformerSize; i++ {
		var attn transformer
		idx = attn.loadFrom(layers, idx)
		m.attn = append(m.attn, &attn)
	}
	m.flatten = layers[idx].(*layer.Flatten)
	idx++
	m.relu = layers[idx].(*activation.ReLU)
	idx++
	m.output = layers[idx].(*layer.Dense)
}
