package model

import (
	"fmt"
	"io"
	_ "net/http/pprof"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/loss"
	pkgmodel "github.com/lwch/tnn/nn/model"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
)

const (
	statusTrain = iota
	statusEvaluate
)

// Model 模型
type Model struct {
	layers   []layer.Layer
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
	init := initializer.NewXavierUniform(1)
	for i := 0; i < transformerSize; i++ {
		m.addTransformerLayer(init, i)
	}
	m.layers = append(m.layers, activation.NewGeLU())
	output := layer.NewDense(len(m.vocabs), init)
	output.SetName("output")
	m.layers = append(m.layers, output)
}

func (m *Model) addTransformerLayer(init initializer.Initializer, i int) {
	attention := layer.NewSelfAttention(paddingSize, embeddingDim, head, init)
	attention.SetName(fmt.Sprintf("transformer%d_attention", i))
	m.layers = append(m.layers, attention)
	m.layers = append(m.layers, layer.NewNor())
	dense1 := layer.NewDense(unitSize*4, init)
	dense1.SetName(fmt.Sprintf("transformer%d_dense1", i))
	m.layers = append(m.layers, dense1)
	m.layers = append(m.layers, activation.NewGeLU())
	dense2 := layer.NewDense(unitSize, init)
	dense2.SetName(fmt.Sprintf("transformer%d_dense2", i))
	m.layers = append(m.layers, dense2)
	m.layers = append(m.layers, layer.NewNor())
}

// getParams 获取需要更新的参数列表
func (m *Model) getParams() ([]*params.Params, []string) {
	var ret []*params.Params
	var names []string
	for _, layer := range m.layers {
		params := layer.Params()
		if params.IsEmpty() {
			continue
		}
		ret = append(ret, params)
		names = append(names, layer.Name())
	}
	return ret, names
}

// zeroGrads 清空梯度
func (m *Model) zeroGrads(paramList []*params.Params) {
	for _, params := range paramList {
		params.Range(func(_ string, p *tensor.Tensor) {
			p.ZeroGrad()
		})
	}
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
	net.Set(m.layers...)
	err := pkgmodel.New(&net, m.loss,
		optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)).
		Save(filepath.Join(m.modelDir, "couplet.model"))
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
