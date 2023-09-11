package net

import (
	"testing"

	"github.com/lwch/tnn/nn/layer"
)

func TestSave(t *testing.T) {
	var net Net
	net.Add(layer.NewLinear(2, 3))
	err := net.Save("test.model", false)
	if err != nil {
		t.Fatal(err)
	}
}
