package optimizer

import (
	"fmt"
	"math"

	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/internal/utils"
	"github.com/lwch/tnn/nn/params"
	"gonum.org/v1/gonum/mat"
)

type Adagrad struct {
	*base
	epsilon float64

	init bool
	g    []*params.Params
}

func NewAdagrad(lr, weightDecay, epsilon float64) *Adagrad {
	var adagrad Adagrad
	adagrad.base = new("adagrad", lr, weightDecay, adagrad.compute)
	adagrad.epsilon = epsilon
	return &adagrad
}

func (adagrad *Adagrad) initParams(grads []*params.Params) {
	if adagrad.init {
		return
	}
	adagrad.g = make([]*params.Params, len(grads))
	for i := 0; i < len(grads); i++ {
		ps := grads[i]
		adagrad.g[i] = params.New()
		ps.Range(func(name string, dense mat.Matrix) {
			rows, cols := dense.Dims()
			adagrad.g[i].Init(name, rows, cols)
		})
	}
	adagrad.init = true
}

func (adagrad *Adagrad) compute(grads []*params.Params) []*params.Params {
	if !adagrad.init {
		adagrad.initParams(grads)
	}
	for i := 0; i < len(grads); i++ {
		ps := grads[i]
		ps.Range(func(name string, dense mat.Matrix) {
			paramG := adagrad.g[i].Get(name)
			var deltaG mat.Dense
			deltaG.Apply(func(i, j int, v float64) float64 {
				return math.Pow(v, 2)
			}, dense)
			paramG.(utils.DenseAdd).Add(paramG, &deltaG)
			var adjust mat.Dense
			adjust.Apply(func(i, j int, v float64) float64 {
				return -adagrad.lr / math.Pow((v+adagrad.epsilon), 0.5)
			}, paramG)
			dense.(utils.DenseMulElem).MulElem(&adjust, dense)
		})
	}
	return grads
}

func (adagrad *Adagrad) Save() *pb.Optimizer {
	ret := adagrad.base.Save()
	ret.Name = "adagrad"
	ret.Params = make(map[string]float64)
	ret.Params["epsilon"] = adagrad.epsilon
	return ret
}

func (adagrad *Adagrad) Print() {
	adagrad.base.Print()
	fmt.Println("  - epsilon:", adagrad.epsilon)
}
