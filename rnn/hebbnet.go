package main

import (
	"github.com/unixpickle/hebbnet"
	"github.com/unixpickle/weakai/rnn"
)

// NewHebbNet creates a Hebbian network model.
// The arguments are similar to NewLSTM.
func NewHebbNet(inSize, hiddenSize, hiddenCount, outHiddenSize, outCount int) *Model {
	return NewBlockModel(inSize, hiddenSize, hiddenCount, outHiddenSize,
		outCount, func(i, h int) rnn.Block {
			res := hebbnet.NewDenseLayer(i, h, true)
			res.UseActivation = true
			return res
		})
}
