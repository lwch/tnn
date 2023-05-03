package layer

import (
	"fmt"

	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/internal/utils"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/params"
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
	}, init)
	return &layer
}

func LoadDense(name string, params map[string]*pb.Dense, _ map[string]*pb.Dense) Layer {
	var layer Dense
	layer.base = new("dense", nil, nil)
	layer.name = name
	layer.base.loadParams(params)
	return &layer
}

func (layer *Dense) Forward(input mat.Matrix, _ bool) (context []mat.Matrix, output mat.Matrix) {
	if !layer.hasInit {
		layer.mInit.Lock()
		shapeW := layer.shapes["w"]
		_, shapeW.M = input.Dims()
		layer.shapes["w"] = shapeW
		layer.mInit.Unlock()
		layer.initParams()
	}
	var ret mat.Dense
	ret.Mul(input, layer.params.Get("w"))
	b := layer.params.Get("b").(utils.DenseRowView).RowView(0)
	rows, _ := ret.Dims()
	for i := 0; i < rows; i++ {
		row := ret.RowView(i)
		row.(utils.AddVec).AddVec(row, b)
	}
	return []mat.Matrix{input}, &ret
}

func (layer *Dense) Backward(context []mat.Matrix, grad mat.Matrix) (valueGrad mat.Matrix, paramsGrad *params.Params) {
	paramsGrad = params.New()
	layer.mInit.Lock()
	sw := layer.shapes["w"]
	sb := layer.shapes["b"]
	layer.mInit.Unlock()
	dw := paramsGrad.Init("w", sw.M, sw.N)
	db := paramsGrad.Init("b", sb.M, sb.N)

	dw.(utils.DenseMul).Mul(context[0].T(), grad)
	db0 := db.(utils.DenseRowView).RowView(0)
	rows, _ := grad.Dims()
	for i := 0; i < rows; i++ {
		db0.(utils.AddVec).AddVec(db0, grad.(utils.DenseRowView).RowView(i))
	}
	db0.(utils.ScaleVec).ScaleVec(1/float64(rows), db0)

	var ret mat.Dense
	w := layer.params.Get("w")
	ret.Mul(grad, w.T())
	return &ret, paramsGrad
}

func (layer *Dense) Print() {
	layer.base.Print()
	_, cnt := layer.params.Get("w").Dims()
	fmt.Println("    Output Count:", cnt)
	fmt.Println("    Params:")
	layer.params.Range(func(name string, dense mat.Matrix) {
		rows, cols := dense.Dims()
		fmt.Println("      - "+name+":", fmt.Sprintf("%dx%d", rows, cols))
	})
}
