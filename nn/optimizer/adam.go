package optimizer

import (
	"fmt"
	"math"

	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/internal/utils"
	"github.com/lwch/tnn/nn/params"
	"gonum.org/v1/gonum/mat"
)

type Adam struct {
	*base
	beta1, beta2 float64
	epsilon      float64

	init bool
	t    int
	m, v []*params.Params
}

func NewAdam(lr, weightDecay, beta1, beta2, epsilon float64) *Adam {
	var adam Adam
	adam.base = new("adam", lr, weightDecay, adam.compute)
	adam.beta1 = beta1
	adam.beta2 = beta2
	adam.epsilon = epsilon
	return &adam
}

func (adam *Adam) initParams(grads []*params.Params) {
	if adam.init {
		return
	}
	adam.m = make([]*params.Params, len(grads))
	adam.v = make([]*params.Params, len(grads))
	for i := 0; i < len(grads); i++ {
		ps := grads[i]
		adam.m[i] = params.New()
		adam.v[i] = params.New()
		ps.Range(func(name string, dense mat.Matrix) {
			rows, cols := dense.Dims()
			adam.m[i].Init(name, rows, cols)
			adam.v[i].Init(name, rows, cols)
		})
	}
	adam.init = true
}

func (adam *Adam) compute(grads []*params.Params) []*params.Params {
	if !adam.init {
		adam.initParams(grads)
	}
	adam.t++
	for i := 0; i < len(grads); i++ {
		ps := grads[i]
		ps.Range(func(name string, dense mat.Matrix) {
			paramM := adam.m[i].Get(name)
			paramV := adam.v[i].Get(name)
			var deltaM, deltaV mat.Dense
			deltaM.Apply(func(i, j int, v float64) float64 {
				return (1 - adam.beta1) * (v - paramM.At(i, j))
			}, dense)
			deltaV.Apply(func(i, j int, v float64) float64 {
				return (1 - adam.beta2) * (math.Pow(v, 2) - paramV.At(i, j))
			}, dense)
			paramM.(utils.DenseAdd).Add(paramM, &deltaM)
			paramV.(utils.DenseAdd).Add(paramV, &deltaV)

			deltaM.Apply(func(i, j int, v float64) float64 {
				return v / (1 - math.Pow(adam.beta1, float64(adam.t)))
			}, paramM)
			var tmp mat.Dense
			tmp.Apply(func(i, j int, v float64) float64 {
				v = v / (1 - math.Pow(adam.beta2, float64(adam.t)))
				return math.Pow(v, 0.5) + adam.epsilon
			}, paramV)
			dense.(utils.DenseScale).Scale(-adam.lr, &deltaM)
			dense.(utils.DenseDivElem).DivElem(dense, &tmp)
		})
	}
	return grads
}

func (adam *Adam) Save() *pb.Optimizer {
	ret := adam.base.Save()
	ret.Name = "adam"
	ret.Params = make(map[string]float64)
	ret.Params["beta1"] = adam.beta1
	ret.Params["beta2"] = adam.beta2
	ret.Params["epsilon"] = adam.epsilon
	return ret
}

func (adam *Adam) Print() {
	adam.base.Print()
	fmt.Println("  - beta1:", adam.beta1)
	fmt.Println("  - beta2:", adam.beta2)
	fmt.Println("  - epsilon:", adam.epsilon)
}
