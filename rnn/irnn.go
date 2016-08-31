package main

import "github.com/unixpickle/weakai/rnn"

// NewIRNN creates an identity RNN model.
// The arguments are similar to NewLSTM.
// There is an extra argument for the IRNN identity
// scale hyperparameter.
func NewIRNN(inSize, hiddenSize, hiddenCount, outHiddenSize, outCount int,
	idScale float64) *Model {
	return NewBlockModel(inSize, hiddenSize, hiddenCount, outHiddenSize,
		outCount, func(i, h int) rnn.Block {
			return rnn.NewIRNN(i, h, idScale)
		})
}
