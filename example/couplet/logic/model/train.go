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

	"github.com/lwch/gotorch/optimizer"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/runtime"
	"github.com/lwch/tnn/example/couplet/logic/feature"
	"github.com/lwch/tnn/example/couplet/logic/sample"
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

	m.optimizer = optimizer.NewAdam(optimizer.WithAdamLr(lr))
	// optimizer := optimizer.NewSGD(lr, 0)

	go m.showProgress()

	begin := time.Now()
	for i := 0; i < epoch; i++ {
		m.epoch = i + 1
		loss := m.trainEpoch()
		if i%10 == 0 {
			m.showModelInfo()
			fmt.Printf("train %d, cost=%s, loss=%e\n",
				i+1, time.Since(begin).String(),
				loss)
			m.save()
		}
		storage.GC()
	}
	m.save()
}

func (m *Model) trainWorker(samples []pair) float64 {
	x := make([]float32, 0, len(samples)*unitSize)
	y := make([]float32, 0, len(samples)*embeddingDim)
	paddingMask := make([][]bool, 0, batchSize)
	batchSize := 0
	for _, s := range samples {
		xTrain, zTrain, pm := sample.Build(s.x, s.y, paddingSize, m.embedding, m.vocabs)
		x = append(x, xTrain...)
		y = append(y, zTrain...)
		paddingMask = append(paddingMask, pm)
		batchSize++
	}
	xIn := tensor.FromFloat32(storage, x, int64(batchSize), paddingSize, embeddingDim)
	zOut := tensor.FromFloat32(storage, y, int64(batchSize), int64(len(m.vocabs)))
	// pred := m.forward(xIn, buildPaddingMasks(paddingMask), true)
	pred := m.forward(xIn, nil, true)
	loss := lossFunc(pred, zOut)
	loss.Backward()
	m.current.Add(uint64(len(samples)))
	return loss.Value()
}

func (m *Model) trainBatch(b []batch) float64 {
	var wg sync.WaitGroup
	wg.Add(len(b))
	var sum float64
	for i := 0; i < len(b); i++ {
		go func(i int) {
			defer wg.Done()
			sum += m.trainWorker(b[i].data)
		}(i)
	}
	wg.Wait()
	m.optimizer.Step(m.params())
	return sum / float64(len(b))
}

// trainEpoch 运行一个批次
func (m *Model) trainEpoch() float64 {
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
	workerCount := rt.NumCPU() * 2
	// workerCount = 1

	var batches []batch
	i := 0
	for {
		if i >= len(idx) {
			break
		}
		var b batch
		for len(b.data) < batchSize {
			if i >= len(idx) {
				break
			}
			idx := idx[i]
			i++
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
				m.current.Add(1)
				continue
			}
			b.append(pair{append(dx, dy...), dz})
		}
		if b.size() > 0 {
			batches = append(batches, b)
		}
	}

	var trainBatch []batch
	var sum float64
	var size float64
	for _, b := range batches {
		trainBatch = append(trainBatch, b)
		if len(trainBatch) >= workerCount {
			sum += m.trainBatch(trainBatch)
			trainBatch = trainBatch[:0]
			size++
		}
	}
	if len(trainBatch) > 0 {
		sum += m.trainBatch(trainBatch)
		size++
	}
	return sum / size
}

func (m *Model) showModelInfo() {
	// list, names := m.getParams()
	// table := tablewriter.NewWriter(os.Stdout)
	// defer table.Render()
	// table.SetHeader([]string{"name", "count"})
	// var total int
	// for i, params := range list {
	// 	var cnt int
	// 	params.Range(func(_ string, p *tensor.Tensor) {
	// 		rows, cols := p.Dims()
	// 		cnt += rows * cols
	// 	})
	// 	table.Append([]string{names[i], fmt.Sprintf("%d", cnt)})
	// 	total += cnt
	// }
	// table.Append([]string{"total", fmt.Sprintf("%d", total)})
}
