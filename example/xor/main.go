package main

import (
	"fmt"
	"math/rand"
	"time"
	"tnn/internal/initializer"
	"tnn/internal/nn/layer"
	"tnn/internal/nn/loss"
	"tnn/internal/nn/model"
	"tnn/internal/nn/net"
	"tnn/internal/nn/optimizer"

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

	row := func(i int) (*mat.Dense, *mat.Dense) {
		return mat.NewDense(1, 2, input.RawRowView(i)),
			mat.NewDense(1, 1, output.RawRowView(i))
	}

	const hidden = 16
	const inputCols = 2
	const outputCols = 1

	rand.Seed(time.Now().UnixNano())

	var net net.Net
	net.Add(layer.NewLinear(1, inputCols, hidden, initializer.NewNormal(100, 1)))
	net.Add(layer.NewSigmoid())
	net.Add(layer.NewLinear(1, hidden, outputCols, initializer.NewNormal(50, 1)))
	loss := loss.NewMSE()
	optimizer := optimizer.NewSGD(lr, 0.9)
	m := model.New(&net, loss, optimizer)
	for i := 0; i < epoch; i++ {
		dInput, dOutput := row(rand.Intn(4))
		loss := m.Train(dInput, dOutput)
		if i%1000 == 0 {
			fmt.Printf("Epoch: %d, Loss: %f\n", i, loss)
		}
	}
	for i := 0; i < 4; i++ {
		dInput, _ := row(i)
		pred := m.Predict(dInput)
		fmt.Println(mat.Formatted(pred))
	}
}
