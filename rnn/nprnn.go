package main

import "github.com/unixpickle/weakai/rnn"

// NewNPRNN creates an np-RNN model.
// The arguments are similar to NewLSTM.
func NewNPRNN(inSize, hiddenSize, hiddenCount, outHiddenSize, outCount int) *Model {
	return NewBlockModel(inSize, hiddenSize, hiddenCount, outHiddenSize,
		outCount, func(i, h int) rnn.Block {
			return rnn.NewNPRNN(i, h)
		})
}
