package matrix

import "fmt"

// Matrix is a matrix.
type Matrix struct {
	m, n int
	data [][]float64
}

// New returns a new matrix with m rows and n columns.
func New(m, n int) *Matrix {
	data := make([][]float64, m)
	for i := range data {
		data[i] = make([]float64, n)
	}
	return &Matrix{m, n, data}
}

// Print prints the matrix.
func (m *Matrix) Print() {
	fmt.Println("Matrix: ==>")
	for _, row := range m.data {
		fmt.Println(row)
	}
	fmt.Println("============================")
}

// Clone returns a copy of the matrix.
func (m *Matrix) Clone() *Matrix {
	ret := New(m.m, m.n)
	for i := 0; i < m.m; i++ {
		for j := 0; j < m.n; j++ {
			ret.data[i][j] = m.data[i][j]
		}
	}
	return ret
}

// T returns the transpose of the matrix.
func (m *Matrix) T() *Matrix {
	ret := New(m.n, m.m)
	for i := 0; i < m.m; i++ {
		for j := 0; j < m.n; j++ {
			ret.data[j][i] = m.data[i][j]
		}
	}
	return ret
}

// Fill fills the matrix with n.
func (m *Matrix) Fill(n float64) {
	for i := 0; i < m.m; i++ {
		for j := 0; j < m.n; j++ {
			m.data[i][j] = n
		}
	}
}
