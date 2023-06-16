package model

import (
	"fmt"
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
	trainX := feature.LoadData(filepath.Join(sampleDir, "in.txt"), m.vocabsIdx, -1)
	trainY := feature.LoadData(filepath.Join(sampleDir, "out.txt"), m.vocabsIdx, -1)
	for i := 0; i < len(trainX); i++ {
		m.samples = append(m.samples, sample.New(trainX[i], trainY[i]))
	}

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

	m.total = len(m.samples)

	m.optimizer = optimizer.NewAdam(optimizer.WithAdamLr(lr))
	// optimizer := optimizer.NewSGD(lr, 0)

	go m.showProgress()

	begin := time.Now()
	for i := 0; i < epoch; i++ {
		m.epoch = i + 1
		lr := m.scheduler.Get()
		loss := m.trainEpoch()
		// m.optimizer.SetLr(m.scheduler.Get())
		m.optimizer.Step(m.params())
		m.scheduler.Step()
		m.save()
		fmt.Printf("train %d, cost=%s, lr=%f, loss=%f\n",
			i+1, time.Since(begin).String(),
			lr, loss)
		if i == 0 {
			m.showModelInfo()
		}
	}
	m.save()
}

func (m *Model) trainWorker(samples []*sample.Sample) float64 {
	x := make([]float32, 0, len(samples)*paddingSize*embeddingDim)
	y := make([]int64, 0, len(samples)*paddingSize)
	padding := make([]int, 0, len(samples))
	for _, s := range samples {
		xTrain, yTrain, p := s.Embedding(paddingSize, m.embedding)
		x = append(x, xTrain...)
		y = append(y, yTrain...)
		padding = append(padding, p)
	}
	xIn := tensor.FromFloat32(storage, x, int64(len(samples)), paddingSize, embeddingDim)
	yOut := tensor.FromInt64(storage, y, int64(len(samples)), paddingSize)
	pred := m.forward(xIn, padding, true)
	pred = pred.Permute(0, 2, 1)
	loss := lossFunc(pred, yOut)
	loss.Backward()
	m.current.Add(uint64(len(samples)))
	return loss.Value()
}

func (m *Model) trainBatch(b []batch) float64 {
	var wg sync.WaitGroup
	wg.Add(len(b))
	var sum float64
	for i := 0; i < len(b); i++ {
		go func(samples []*sample.Sample) {
			defer wg.Done()
			sum += m.trainWorker(samples)
		}(b[i].data)
	}
	wg.Wait()
	storage.GC()
	return sum / float64(len(b))
}

// trainEpoch 运行一个批次
func (m *Model) trainEpoch() float64 {
	m.status = statusTrain
	m.current.Store(0)

	// 生成索引序列
	idx := make([]int, len(m.samples))
	for i := 0; i < len(idx); i++ {
		idx[i] = i
	}
	// rand.Shuffle(len(idx), func(i, j int) {
	// 	idx[i], idx[j] = idx[j], idx[i]
	// })

	// 创建训练协程并行训练
	workerCount := rt.NumCPU() * 2
	// workerCount = 1

	var batches []batch
	for i := 0; i < len(m.samples); i += batchSize {
		var b batch
		for j := 0; j < batchSize; j++ {
			if i+j >= len(m.samples) {
				break
			}
			b.append(m.samples[idx[i+j]])
		}
		batches = append(batches, b)
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
	table := tablewriter.NewWriter(os.Stdout)
	defer table.Render()
	table.SetHeader([]string{"name", "count"})
	var total int64
	for _, attn := range m.attn {
		cnt := paramSize(attn.attn.Params())
		total += cnt
		table.Append([]string{attn.attn.Name(), fmt.Sprintf("%d", cnt)})
		cnt = paramSize(attn.dense.Params())
		total += cnt
		table.Append([]string{attn.dense.Name(), fmt.Sprintf("%d", cnt)})
		cnt = paramSize(attn.output.Params())
		total += cnt
		table.Append([]string{attn.output.Name(), fmt.Sprintf("%d", cnt)})
	}
	cnt := paramSize(m.output.Params())
	total += cnt
	table.Append([]string{m.output.Name(), fmt.Sprintf("%d", cnt)})
	table.Append([]string{"total", fmt.Sprintf("%d", total)})
}

func paramSize(params map[string]*tensor.Tensor) int64 {
	var ret int64
	for _, p := range params {
		size := int64(1)
		for _, s := range p.Shapes() {
			size *= s
		}
		ret += size
	}
	return ret
}
