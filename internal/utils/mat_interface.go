package utils

import "gonum.org/v1/gonum/mat"

type DenseAdd interface {
	Add(a, b mat.Matrix)
}

type DenseSub interface {
	Sub(a, b mat.Matrix)
}

type DenseMul interface {
	Mul(a, b mat.Matrix)
}

type DenseDivElem interface {
	DivElem(a, b mat.Matrix)
}

type DenseScale interface {
	Scale(f float64, a mat.Matrix)
}

type DenseCopy interface {
	Copy(a mat.Matrix) (r, c int)
}

type DenseApply interface {
	Apply(fn func(i, j int, v float64) float64, a mat.Matrix)
}

type DenseRowView interface {
	RowView(i int) mat.Vector
}

type DenseSlice interface {
	Slice(i, k, j, l int) mat.Matrix
}

type DenseSet interface {
	Set(i, j int, v float64)
}

// vector
type AddVec interface {
	AddVec(a, b mat.Vector)
}
