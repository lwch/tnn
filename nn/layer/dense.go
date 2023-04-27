package layer

import (
	"fmt"
	"sync"

	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/internal/utils"
	"github.com/lwch/tnn/nn/initializer"
	"gonum.org/v1/gonum/mat"
)

type Dense struct {
	*base
}

func NewDense(output int, init initializer.Initializer) *Dense {
	var layer Dense
	layer.base = new("dense", map[string]Shape{
		"w": {NoneShape, output}, // rows reshape from input
		"b": {1, output},         // rows reshape from input
	}, init, layer.forward, layer.backward)
	return &layer
}

func LoadDense(name string, params map[string]*pb.Dense, _ map[string]*pb.Dense) Layer {
	var layer Dense
	layer.base = new("dense", nil, nil, layer.forward, layer.backward)
	layer.name = name
	layer.base.loadParams(params)
	return &layer
}

func (layer *Dense) forward(input mat.Matrix) mat.Matrix {
	if !layer.hasInit {
		shapeW := layer.shapes["w"]
		_, shapeW.M = input.Dims()
		layer.shapes["w"] = shapeW
		layer.initParams()
	}
	var ret mat.Dense
	ret.Mul(input, layer.params["w"])
	b := layer.params["b"].(utils.DenseRowView).RowView(0)
	rows, _ := ret.Dims()
	var wg sync.WaitGroup
	wg.Add(rows)
	for i := 0; i < rows; i++ {
		go func(i int) {
			defer wg.Done()
			row := ret.RowView(i)
			row.(utils.AddVec).AddVec(row, b)
		}(i)
	}
	wg.Wait()
	return &ret
}

func (layer *Dense) backward(grad mat.Matrix) mat.Matrix {
	dw := layer.context["w"]
	db := layer.context["b"]

	dw.(utils.DenseMul).Mul(layer.input.T(), grad)
	db0 := db.(utils.DenseRowView).RowView(0)
	rows, _ := grad.Dims()
	var wg sync.WaitGroup
	wg.Add(rows)
	for i := 0; i < rows; i++ {
		go func(i int) {
			defer wg.Done()
			db0.(utils.AddVec).AddVec(db0, grad.(utils.DenseRowView).RowView(i))
		}(i)
	}
	wg.Wait()
	db0.(utils.ScaleVec).ScaleVec(1/float64(rows), db0)

	var ret mat.Dense
	w := layer.params["w"]
	ret.Mul(grad, w.T())
	return &ret
}

func (layer *Dense) Print() {
	layer.base.Print()
	_, cnt := layer.params["w"].Dims()
	fmt.Println("    Output Count:", cnt)
	fmt.Println("    Params:")
	for name, dense := range layer.params {
		rows, cols := dense.Dims()
		fmt.Println("      - "+name+":", fmt.Sprintf("%dx%d", rows, cols))
	}
}
