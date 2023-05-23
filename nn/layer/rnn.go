package layer

import (
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/tensor"
)

type Rnn struct {
	*base
	featureSize, times int
	hidden             int
}

func NewRnn(featureSize, times, hidden int, init initializer.Initializer) Layer {
	var layer Rnn
	layer.base = new("rnn", map[string]Shape{
		"dw": {featureSize, hidden},
		"db": {1, hidden},
		"hw": {hidden, hidden},
		"hb": {1, hidden},
	}, init)
	layer.featureSize = featureSize
	layer.times = times
	layer.hidden = hidden
	return &layer
}

func (layer *Rnn) Forward(input *tensor.Tensor, isTraining bool) *tensor.Tensor {
	if !layer.hasInit {
		layer.initParams()
	}
	dw := layer.params.Get("dw")
	db := layer.params.Get("db")
	hw := layer.params.Get("hw")
	hb := layer.params.Get("hb")
	batchSize, _ := input.Dims()
	state := tensor.New(nil, batchSize, layer.hidden)
	for t := layer.times - 1; t >= 0; t-- {
		start := t * layer.featureSize
		t := input.Slice(0, batchSize, start, start+layer.featureSize)
		l1 := t.Mul(dw).AddVector(db)
		l2 := state.Mul(hw).AddVector(hb)
		state = l1.Add(l2).Tanh()
	}
	return state
}
