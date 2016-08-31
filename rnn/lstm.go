package main

import "github.com/unixpickle/weakai/rnn"

// NewLSTM creates an LSTM-based model.
// The hiddenSize specifies the number of LSTM units
// in each hidden recurrent layer, and hiddenCount
// determines the number of hidden recurrent layers.
// The outHiddenSize argument specifies how many hidden
// neurons to use in the non-recurrent output layer.
func NewLSTM(inSize, hiddenSize, hiddenCount, outHiddenSize, outCount int) *Model {
	return NewBlockModel(inSize, hiddenSize, hiddenCount, outHiddenSize,
		outCount, func(i, h int) rnn.Block {
			return rnn.NewLSTM(i, h)
		})
}
