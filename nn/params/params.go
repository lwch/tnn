package params

import (
	"fmt"
	"sync"

	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/internal/utils"
	"gonum.org/v1/gonum/mat"
)

type Params struct {
	m    sync.RWMutex
	data map[string]mat.Matrix
}

func New() *Params {
	return &Params{data: make(map[string]mat.Matrix)}
}

func (params *Params) Copy(ps *Params) {
	params.m.Lock()
	defer params.m.Unlock()
	ps.m.RLock()
	defer ps.m.RUnlock()
	if params.data == nil {
		params.data = make(map[string]mat.Matrix)
	}
	for name, value := range ps.data {
		var dense mat.Dense
		dense.CloneFrom(value)
		params.data[name] = &dense
	}
}

func (params *Params) Add(ps *Params) {
	if ps == nil {
		return
	}
	params.m.Lock()
	defer params.m.Unlock()
	ps.m.RLock()
	defer ps.m.RUnlock()
	for name, grad := range ps.data {
		p := params.data[name]
		if p == nil {
			continue
		}
		p.(utils.DenseAdd).Add(p, grad)
	}
}

func (params *Params) Scale(n float64) {
	if params == nil {
		return
	}
	params.m.Lock()
	defer params.m.Unlock()
	for _, dense := range params.data {
		dense.(utils.DenseScale).Scale(n, dense)
	}
}

func (params *Params) Range(fn func(name string, dense mat.Matrix)) {
	if params == nil {
		return
	}
	data := make(map[string]mat.Matrix, len(params.data))
	params.m.RLock()
	for k, v := range params.data {
		data[k] = v
	}
	params.m.RUnlock()
	for name, dense := range data {
		fn(name, dense)
	}
}

func (params *Params) Init(name string, rows, cols int) mat.Matrix {
	ret := mat.NewDense(rows, cols, nil)
	params.m.Lock()
	defer params.m.Unlock()
	params.data[name] = ret
	return ret
}

func (params *Params) InitWithData(name string, rows, cols int, data []float64) {
	params.m.Lock()
	defer params.m.Unlock()
	params.data[name] = mat.NewDense(rows, cols, data)
}

func (params *Params) Get(name string) mat.Matrix {
	params.m.RLock()
	defer params.m.RUnlock()
	return params.data[name]
}

func (params *Params) Print() {
	if len(params.data) == 0 {
		return
	}
	params.m.RLock()
	defer params.m.RUnlock()
	fmt.Println("============ params ==================")
	for name, dense := range params.data {
		fmt.Println(name)
		fmt.Println(mat.Formatted(dense))
	}
	fmt.Println("======================================")
}

func (params *Params) Size() int {
	return len(params.data)
}

func (params *Params) Load(from map[string]*pb.Dense) {
	params.m.Lock()
	defer params.m.Unlock()
	params.data = make(map[string]mat.Matrix, len(from))
	for name, param := range from {
		params.data[name] = mat.NewDense(int(param.GetRows()), int(param.GetCols()), param.GetData())
	}
}
