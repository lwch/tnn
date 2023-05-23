package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
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

func LoadRnn(name string, params map[string]*pb.Dense, args map[string]*pb.Dense) Layer {
	arr := args["params"].GetData()
	var layer Rnn
	layer.featureSize = int(arr[0])
	layer.times = int(arr[1])
	layer.hidden = int(arr[2])
	layer.base = new("rnn", nil, nil)
	layer.name = name
	layer.base.loadParams(params)
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

func (layer *Rnn) Args() map[string]*mat.VecDense {
	return map[string]*mat.VecDense{
		"params": mat.NewVecDense(3, []float64{
			float64(layer.featureSize),
			float64(layer.times),
			float64(layer.hidden)}),
	}
}
