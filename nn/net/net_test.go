package net

import (
	"testing"

	"github.com/lwch/tnn/nn/layer"
)

func TestSave(t *testing.T) {
	var net Net
	net.Add(layer.NewLinear("linear", 2, 3))
	err := net.Save("test.model")
	if err != nil {
		t.Fatal(err)
	}
}

func TestLoad(t *testing.T) {
	var net Net
	err := net.Load("test.model")
	if err != nil {
		t.Fatal(err)
	}
}
