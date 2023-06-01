package model

import (
	"fmt"
	"io"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path/filepath"
	"time"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/example/couplet/logic/feature"
	"github.com/lwch/tnn/nn/loss"
	pkgmodel "github.com/lwch/tnn/nn/model"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"github.com/lwch/tnn/nn/tensor"
)

// Train 训练模型
func (m *Model) Train(sampleDir, modelDir string) {
	go func() { // for pprof
		http.ListenAndServe(":8888", nil)
	}()

	m.modelDir = modelDir

	// 加载样本
	m.vocabs, m.vocabsIdx = feature.LoadVocab(filepath.Join(sampleDir, "vocabs"))
	m.trainX = feature.LoadData(filepath.Join(sampleDir, "in.txt"), m.vocabsIdx, -1)
	m.trainY = feature.LoadData(filepath.Join(sampleDir, "out.txt"), m.vocabsIdx, -1)
	m.copyVocabs(filepath.Join(sampleDir, "vocabs"))

	// 加载embedding
	if _, err := os.Stat(filepath.Join(modelDir, "embedding")); os.IsNotExist(err) {
		m.buildEmbedding(filepath.Join(modelDir, "embedding"))
	}
	m.embedding = m.loadEmbedding(filepath.Join(modelDir, "embedding"))

	m.build()

	m.total = len(m.trainX) * paddingSize

	// loss := loss.NewSoftmax()
	m.loss = loss.NewMSE()
	m.optimizer = optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)
	// optimizer := optimizer.NewSGD(lr, 0)

	go m.showProgress()
	go m.update()

	begin := time.Now()
	for i := 0; i < epoch; i++ {
		m.epoch = i + 1
		m.trainEpoch()
		if i%10 == 0 {
			list, names := m.getParams()
			for i, params := range list {
				var cnt int
				params.Range(func(_ string, p *tensor.Tensor) {
					rows, cols := p.Dims()
					cnt += rows * cols
				})
				fmt.Printf("%s: %d\n", names[i], cnt)
			}
			fmt.Printf("train %d, cost=%s, loss=%.05f\n",
				i+1, time.Since(begin).String(),
				m.avgLoss())
		}
	}
	m.save()
}

// update 梯度更新，每隔1分钟计算一次
func (m *Model) update() {
	tk := time.NewTicker(time.Minute)
	defer tk.Stop()
	for {
		select {
		case <-tk.C:
		case <-m.chUpdate:
		}
		params, _ := m.getParams()
		m.optimizer.Update(params)
		m.save()
		m.zeroGrads(params)
		fmt.Println("params updated")
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
