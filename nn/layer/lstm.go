package layer

import (
	"github.com/lwch/tnn/internal/math"
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
)

type Lstm struct {
	*base
	featureSize, steps int
	hidden             int
}

func NewLstm(featureSize, steps, hidden int, init initializer.Initializer) Layer {
	var layer Lstm
	layer.base = new("lstm", map[string]Shape{
		// It
		"Wii": {featureSize, hidden},
		"Bii": {NoneShape, 1},
		"Whi": {hidden, hidden},
		"Bhi": {NoneShape, 1},
		// Ft
		"Wif": {featureSize, hidden},
		"Bif": {NoneShape, 1},
		"Whf": {hidden, hidden},
		"Bhf": {NoneShape, 1},
		// Gt
		"Wig": {featureSize, hidden},
		"Big": {NoneShape, 1},
		"Whg": {hidden, hidden},
		"Bhg": {NoneShape, 1},
		// Ot
		"Wio": {featureSize, hidden},
		"Bio": {NoneShape, 1},
		"Who": {hidden, hidden},
		"Bho": {NoneShape, 1},
	}, init)
	layer.featureSize = featureSize
	layer.steps = steps
	layer.hidden = hidden
	return &layer
}

func LoadLstm(name string, params map[string]*pb.Dense, args map[string]*pb.Dense) Layer {
	arr := args["params"].GetData()
	var layer Lstm
	layer.featureSize = int(arr[0])
	layer.steps = int(arr[1])
	layer.hidden = int(arr[2])
	layer.base = new("lstm", nil, nil)
	layer.name = name
	layer.base.loadParams(params)
	return &layer
}

// Forward https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
func (layer *Lstm) Forward(input *tensor.Tensor, isTraining bool) *tensor.Tensor {
	if !layer.hasInit {
		layer.mInit.Lock()
		shapeBii := layer.shapes["Bii"]
		shapeBhi := layer.shapes["Bhi"]
		shapeBif := layer.shapes["Bif"]
		shapeBhf := layer.shapes["Bhf"]
		shapeBig := layer.shapes["Big"]
		shapeBhg := layer.shapes["Bhg"]
		shapeBio := layer.shapes["Bio"]
		shapeBho := layer.shapes["Bho"]
		shapeBii.M, _ = input.Dims()
		shapeBhi.M, _ = input.Dims()
		shapeBif.M, _ = input.Dims()
		shapeBhf.M, _ = input.Dims()
		shapeBig.M, _ = input.Dims()
		shapeBhg.M, _ = input.Dims()
		shapeBio.M, _ = input.Dims()
		shapeBho.M, _ = input.Dims()
		layer.shapes["Bii"] = shapeBii
		layer.shapes["Bhi"] = shapeBhi
		layer.shapes["Bif"] = shapeBif
		layer.shapes["Bhf"] = shapeBhf
		layer.shapes["Big"] = shapeBig
		layer.shapes["Bhg"] = shapeBhg
		layer.shapes["Bio"] = shapeBio
		layer.shapes["Bho"] = shapeBho
		layer.mInit.Unlock()
		layer.initParams()
	}
	// It
	Wii := layer.params.Get("Wii")
	Bii := layer.params.Get("Bii")
	Whi := layer.params.Get("Whi")
	Bhi := layer.params.Get("Bhi")
	// Ft
	Wif := layer.params.Get("Wif")
	Bif := layer.params.Get("Bif")
	Whf := layer.params.Get("Whf")
	Bhf := layer.params.Get("Bhf")
	// Gt
	Wig := layer.params.Get("Wig")
	Big := layer.params.Get("Big")
	Whg := layer.params.Get("Whg")
	Bhg := layer.params.Get("Bhg")
	// Ot
	Wio := layer.params.Get("Wio")
	Bio := layer.params.Get("Bio")
	Who := layer.params.Get("Who")
	Bho := layer.params.Get("Bho")
	batchSize, _ := input.Dims()
	c := tensor.New(nil, batchSize, layer.hidden)
	h := tensor.New(nil, batchSize, layer.hidden)
	for t := layer.steps - 1; t >= 0; t-- {
		start := t * layer.featureSize
		x := input.Slice(0, batchSize, start, start+layer.featureSize)
		It := math.Sigmoid(x.Mul(Wii).Add(Bii).Add(h.Mul(Whi)).Add(Bhi))
		Ft := math.Sigmoid(x.Mul(Wif).Add(Bif).Add(h.Mul(Whf)).Add(Bhf))
		Gt := x.Mul(Wig).Add(Big).Add(h.Mul(Whg)).Add(Bhg).Tanh()
		Ot := math.Sigmoid(x.Mul(Wio).Add(Bio).Add(h.Mul(Who)).Add(Bho))
		c = Ft.MulElem(c).Add(It.MulElem(Gt))
		h = Ot.MulElem(c.Tanh())
	}
	return h
}

func (layer *Lstm) Args() map[string]*mat.VecDense {
	return map[string]*mat.VecDense{
		"params": mat.NewVecDense(3, []float64{
			float64(layer.featureSize),
			float64(layer.steps),
			float64(layer.hidden)}),
	}
}
