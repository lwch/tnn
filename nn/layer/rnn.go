package layer

import (
	"fmt"
	"math"

	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/internal/utils"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/params"
	"gonum.org/v1/gonum/mat"
)

type Rnn struct {
	*base
	output int
	times  int
}

func NewRnn(times, output int, init initializer.Initializer) *Rnn {
	var layer Rnn
	layer.base = new("rnn", map[string]Shape{
		"w": {output, NoneShape}, // rows reshape from input
		"b": {1, output},         // rows reshape from input
	}, init)
	layer.output = output
	layer.times = times
	return &layer
}

func LoadRnn(name string, params map[string]*pb.Dense, _ map[string]*pb.Dense) Layer {
	var layer Rnn
	layer.base = new("rnn", nil, nil)
	layer.name = name
	layer.base.loadParams(params)
	return &layer
}

func (layer *Rnn) Forward(input mat.Matrix, _ bool) (context []mat.Matrix, output mat.Matrix) {
	batchSize, cols := input.Dims()
	featureSize := cols / layer.times
	if !layer.hasInit {
		layer.mInit.Lock()
		shapeW := layer.shapes["w"]
		shapeW.N = layer.output + featureSize
		layer.shapes["w"] = shapeW
		layer.mInit.Unlock()
		layer.initParams()
	}

	h := mat.NewDense(batchSize, layer.times*layer.output, nil)

	b := layer.params.Get("b").(utils.DenseRowView).RowView(0)

	for t := 0; t < layer.times; t++ {
		start := (t - 1) * layer.output
		if start < 0 {
			start = (layer.times - 1) * layer.output
		}
		dh := h.Slice(0, batchSize, start, start+layer.output)
		start = t * featureSize
		dx := input.(utils.DenseSlice).Slice(0, batchSize, start, start+featureSize)
		var z mat.Dense
		z.Stack(dh.T(), dx.T())
		dz := z.T()
		var a mat.Dense
		a.Mul(dz, layer.params.Get("w").T())
		for i := 0; i < batchSize; i++ {
			row := a.RowView(i)
			row.(utils.AddVec).AddVec(row, b)
		}

		start = t * layer.output
		hRange := h.Slice(0, batchSize, start, start+layer.output)
		hRange.(utils.DenseApply).Apply(func(i, j int, v float64) float64 {
			return math.Tanh(v)
		}, &a)
	}

	return []mat.Matrix{h, input}, h.Slice(0, batchSize, (layer.times-1)*layer.output, layer.times*layer.output)
}

func (layer *Rnn) Backward(context []mat.Matrix, grad mat.Matrix) (valueGrad mat.Matrix, paramsGrad *params.Params) {
	paramsGrad = params.New()
	layer.mInit.Lock()
	sw := layer.shapes["w"]
	sb := layer.shapes["b"]
	layer.mInit.Unlock()
	dw := paramsGrad.Init("w", sw.M, sw.N)
	db := paramsGrad.Init("b", sb.M, sb.N)
	db0 := db.(utils.DenseRowView).RowView(0)

	h := context[0]
	input := context[1]

	batchSize, cols := input.Dims()
	featureSize := cols / layer.times
	dIn := mat.NewDense(batchSize, cols, nil)
	dH := grad

	for t := layer.times - 1; t >= 0; t-- {
		start := t * layer.output
		tmp := h.(utils.DenseSlice).Slice(0, batchSize, start, start+layer.output)
		var da mat.Dense
		da.Apply(func(i, j int, v float64) float64 {
			return dH.At(i, j) * (1 - math.Pow(v, 2))
		}, tmp)

		w := layer.params.Get("w").(utils.DenseSlice).Slice(0, layer.output, layer.output, layer.output+featureSize)
		var m mat.Dense
		m.Mul(&da, w)
		start = t * featureSize
		dIn.Slice(0, batchSize, start, start+featureSize).(utils.DenseCopy).Copy(&m)

		aT := da.T()

		start = t * featureSize
		i := input.(utils.DenseSlice).Slice(0, batchSize, start, start+featureSize)
		var wI mat.Dense
		wI.Mul(aT, i)
		dwRange := dw.(utils.DenseSlice).Slice(0, layer.output, layer.output, layer.output+featureSize)
		dwRange.(utils.DenseAdd).Add(dwRange, &wI)

		start = (t - 1) * layer.output
		if start < 0 {
			start = (layer.times - 1) * layer.output
		}
		hRange := h.(utils.DenseSlice).Slice(0, batchSize, start, start+layer.output)
		var wH mat.Dense
		wH.Mul(aT, hRange)
		dwRange = dw.(utils.DenseSlice).Slice(0, layer.output, 0, layer.output)
		dwRange.(utils.DenseAdd).Add(dwRange, &wH)

		rows, _ := da.Dims()
		for i := 0; i < rows; i++ {
			db0.(utils.AddVec).AddVec(db0, da.RowView(i))
		}

		w = layer.params.Get("w").(utils.DenseSlice).Slice(0, layer.output, 0, layer.output)
		dH.(utils.DenseMul).Mul(&da, w)
	}

	return dIn, paramsGrad
}

func (layer *Rnn) Print() {
	layer.base.Print()
	_, cnt := layer.params.Get("w").Dims()
	fmt.Println("    Output Count:", cnt)
	fmt.Println("    Params:")
	layer.params.Range(func(name string, dense mat.Matrix) {
		rows, cols := dense.Dims()
		fmt.Println("      - "+name+":", fmt.Sprintf("%dx%d", rows, cols))
	})
}

// func (layer *Rnn) pad(dense *mat.Dense) *mat.Dense {
// 	rows, cols := dense.Dims()
// 	ret := mat.NewDense(rows, cols+layer.output, nil)
// 	ret.Slice(0, rows, 0, cols).(utils.DenseCopy).Copy(dense)
// 	return ret
// }
