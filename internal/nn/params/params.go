package params

import "gonum.org/v1/gonum/mat"

type Params map[string]*mat.Dense

func (params *Params) Copy(ps Params) {
	*params = make(Params)
	for name, value := range ps {
		var dense mat.Dense
		dense.CloneFrom(value)
		(*params)[name] = &dense
	}
}

func (params Params) Add(grads *Params) {
	for name, grad := range *grads {
		p := params[name]
		if p == nil {
			continue
		}
		p.Add(p, grad)
	}
}
