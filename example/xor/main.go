package main

import (
	"fmt"
	"tnn/internal/initializer"
	"tnn/internal/nn/layer"
	"tnn/internal/nn/loss"
	"tnn/internal/nn/model"
	"tnn/internal/nn/net"
	"tnn/internal/nn/optimizer"
	"tnn/internal/shape"

	"gonum.org/v1/gonum/mat"
)

const lr = 0.01
const epoch = 10000

func main() {
	input := mat.NewDense(4, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		0, 0,
	})
	output := mat.NewDense(4, 1, []float64{
		0,
		1,
		1,
		0,
	})

	var net net.Net
	net.Add(layer.NewLinear(shape.Shape{M: 4, N: 2}, 1, initializer.NewNormal(0, 1)))
	loss := loss.NewMSE()
	optimizer := optimizer.NewSGD(lr, 0.9)
	m := model.New(&net, loss, optimizer)
	for i := 0; i < epoch; i++ {
		loss := m.Train(input, output)
		if i%100 == 0 {
			fmt.Printf("Epoch: %d, Loss: %f\n", i, loss)
		}
	}
	fmt.Println(m.Predict(input))
}
