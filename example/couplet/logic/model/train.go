package model

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	rt "runtime"
	"sync"
	"time"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/example/couplet/logic/feature"
	"github.com/lwch/tnn/example/couplet/logic/sample"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/optimizer"
	"github.com/lwch/tnn/nn/tensor"
	"github.com/olekukonko/tablewriter"
)

// Train 训练模型
func (m *Model) Train(sampleDir, modelDir string) {
	// go func() { // for pprof
	// 	http.ListenAndServe(":8888", nil)
	// }()

	m.modelDir = modelDir
	runtime.Assert(os.MkdirAll(modelDir, 0755))

	// 加载样本
	m.vocabs, m.vocabsIdx = feature.LoadVocab(filepath.Join(sampleDir, "vocabs"))
	m.trainX = feature.LoadData(filepath.Join(sampleDir, "in.txt"), m.vocabsIdx, -1)
	m.trainY = feature.LoadData(filepath.Join(sampleDir, "out.txt"), m.vocabsIdx, -1)

	// 加载embedding
	if _, err := os.Stat(filepath.Join(modelDir, "embedding")); os.IsNotExist(err) {
		m.buildEmbedding(filepath.Join(modelDir, "embedding"))
	}

	if _, err := os.Stat(filepath.Join(modelDir, "couplet.model")); !os.IsNotExist(err) {
		m.Load(m.modelDir)
	} else {
		m.copyVocabs(filepath.Join(sampleDir, "vocabs"))
		m.embedding = m.loadEmbedding(filepath.Join(modelDir, "embedding"))
		m.build()
	}

	m.total = len(m.trainX) * paddingSize / 2

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
			m.showModelInfo()
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

// trainWorker 训练协程
func (m *Model) trainWorker(ch chan []int) {
	for {
		idx, ok := <-ch
		if !ok {
			return
		}
		x := make([]float64, 0, len(idx)*unitSize)
		y := make([]float64, 0, len(idx)*embeddingDim)
		paddingMask := make([][]bool, 0, batchSize)
		for _, idx := range idx {
			i := math.Floor(float64(idx) / float64(paddingSize/2))
			j := idx % (paddingSize / 2)
			dx := m.trainX[int(i)]
			dy := m.trainY[int(i)]
			dy = append([]int{0}, dy...) // <s> ...
			var dz int
			if j < len(dy) {
				dz = dy[j]
				dy = dy[:j]
			} else {
				dz = -1
			}
			xTrain, zTrain, pm := sample.Build(append(dx, dy...), dz, paddingSize, m.embedding, m.vocabs)
			x = append(x, xTrain...)
			y = append(y, zTrain...)
			paddingMask = append(paddingMask, pm)
		}
		xIn := tensor.New(x, len(idx), unitSize)
		zOut := tensor.New(y, len(idx), len(m.vocabs))
		pred := m.forward(xIn, buildPaddingMasks(paddingMask), true)
		grad := m.loss.Loss(pred, zOut)
		grad.Backward(grad)
		m.current.Add(uint64(len(idx)))
	}
}

// trainEpoch 运行一个批次
func (m *Model) trainEpoch() {
	m.status = statusTrain
	m.current.Store(0)

	// 生成索引序列
	idx := make([]int, len(m.trainX)*paddingSize/2)
	for i := 0; i < len(idx); i++ {
		idx[i] = i
	}
	rand.Shuffle(len(idx), func(i, j int) {
		idx[i], idx[j] = idx[j], idx[i]
	})

	// 创建训练协程并行训练
	workerCount := rt.NumCPU()
	// workerCount = 1

	ch := make(chan []int)
	var wg sync.WaitGroup
	wg.Add(workerCount)
	for i := 0; i < workerCount; i++ {
		go func() {
			defer wg.Done()
			m.trainWorker(ch)
		}()
	}

	for i := 0; i < len(idx); i += batchSize {
		list := make([]int, 0, batchSize)
		for j := 0; j < batchSize; j++ {
			if i+j >= len(idx) {
				break
			}
			list = append(list, idx[i+j])
		}
		ch <- list
	}
	close(ch)
	wg.Wait()

	// 触发梯度更新
	m.chUpdate <- struct{}{}
}

func (m *Model) showModelInfo() {
	list, names := m.getParams()
	table := tablewriter.NewWriter(os.Stdout)
	defer table.Render()
	table.SetHeader([]string{"name", "count"})
	var total int
	for i, params := range list {
		var cnt int
		params.Range(func(_ string, p *tensor.Tensor) {
			rows, cols := p.Dims()
			cnt += rows * cols
		})
		table.Append([]string{names[i], fmt.Sprintf("%d", cnt)})
		total += cnt
	}
	table.Append([]string{"total", fmt.Sprintf("%d", total)})
}
