package tensor

func Ones(rows, cols int) *Tensor {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = 1
	}
	return New(data, rows, cols)
}
