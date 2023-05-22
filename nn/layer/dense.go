package layer

import (
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
)

type Dense struct {
	*base
}

func NewDense(output int, init initializer.Initializer) Layer {
	var layer Dense
	layer.base = new("dense", map[string]Shape{
		"w": {NoneShape, output},
		"b": {1, output},
	}, init)
	return &layer
}

func (layer *Dense) Forward(input *tensor.Tensor, watchList *params.List, isTraining bool) *tensor.Tensor {
	if !layer.hasInit {
		layer.mInit.Lock()
		shapeW := layer.shapes["w"]
		_, shapeW.M = input.Dims()
		layer.shapes["w"] = shapeW
		layer.mInit.Unlock()
		layer.initParams()
	}
	w := layer.params.Get("w")
	b := layer.params.Get("b")
	if watchList != nil {
		watchList.Add(w)
		watchList.Add(b)
	}
	w1 := input.Mul(w)
	w1.SetName(layer.Name() + ".wx")
	w2 := w1.AddVector(b)
	w2.SetName(layer.Name() + ".wx+b")
	return w2
}
