package vector

import "gonum.org/v1/gonum/mat"

type Adder interface {
	Add(a, b mat.Matrix)
}

type Muler interface {
	Mul(a, b mat.Matrix)
}

type Copyer interface {
	Copy(a mat.Matrix) (r, c int)
}

type Applyer interface {
	Apply(fn func(i, j int, v float64) float64, a mat.Matrix)
}

type Scaler interface {
	Scale(f float64, a mat.Matrix)
}
