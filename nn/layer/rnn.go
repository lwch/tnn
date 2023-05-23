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
		"Wih": {featureSize, hidden},
		"Bih": {1, hidden},
		"Whh": {hidden, hidden},
		"Bhh": {1, hidden},
	}, init)
	layer.featureSize = featureSize
	layer.times = times
	layer.hidden = hidden
	return &layer
}

// Forward https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
func (layer *Rnn) Forward(input *tensor.Tensor, isTraining bool) *tensor.Tensor {
	if !layer.hasInit {
		layer.initParams()
	}
	Wih := layer.params.Get("Wih")
	Bih := layer.params.Get("Bih")
	Whh := layer.params.Get("Whh")
	Bhh := layer.params.Get("Bhh")
	batchSize, _ := input.Dims()
	state := tensor.New(nil, batchSize, layer.hidden)
	for t := layer.times - 1; t >= 0; t-- {
		start := t * layer.featureSize
		t := input.Slice(0, batchSize, start, start+layer.featureSize)
		l1 := t.Mul(Wih).AddVector(Bih)
		l2 := state.Mul(Whh).AddVector(Bhh)
		state = l1.Add(l2).Tanh()
	}
	return state
}
