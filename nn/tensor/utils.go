package tensor

func Ones(rows, cols int) *Tensor {
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = 1
	}
	return New(data, rows, cols)
}

func Zeros(rows, cols int) *Tensor {
	data := make([]float32, rows*cols)
	return New(data, rows, cols)
}

func Numbers(rows, cols int, n float32) *Tensor {
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = n
	}
	return New(data, rows, cols)
}
