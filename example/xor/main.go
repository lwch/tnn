package main

import (
	"fmt"
	"math/rand"
	"tnn/internal/initializer"
	"tnn/internal/nn/layer"
	"tnn/internal/nn/loss"
	"tnn/internal/nn/model"
	"tnn/internal/nn/net"
	"tnn/internal/nn/optimizer"

	"gonum.org/v1/gonum/mat"
)

const lr = 0.01
const epoch = 100000

func main() {
	input := mat.NewDense(4, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
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

	initializer := initializer.NewNormal(1, 0.5)

	var net net.Net
	net.Add(
		layer.NewLinear(hidden, initializer),
		layer.NewSigmoid(),
		layer.NewLinear(1, initializer),
	)
	loss := loss.NewMSE()
	optimizer := optimizer.NewSGD(lr, 0)
	// optimizer := optimizer.NewAdam(lr, 0.9, 0.999, 1e-8, 0)
	m := model.New(&net, loss, optimizer)
	for i := 0; i < epoch; i++ {
		dInput, dOutput := row(rand.Intn(4))
		m.Train(dInput, dOutput)
		if i%1000 == 0 {
			loss := m.Loss(dInput, dOutput)
			fmt.Printf("Epoch: %d, Loss: %.05f\n", i, loss)
		}
	}
	for i := 0; i < 4; i++ {
		dInput, _ := row(i)
		pred := m.Predict(dInput)
		fmt.Printf("predict %d xor %d: %f\n",
			int(dInput.At(0, 0)), int(dInput.At(0, 1)),
			pred.At(0, 0))
	}
}
