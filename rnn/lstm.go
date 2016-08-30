package main

import (
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

// NewLSTM creates an LSTM-based model.
// The hiddenSize specifies the number of LSTM units
// in each hidden recurrent layer, and hiddenCount
// determines the number of hidden recurrent layers.
// The outHiddenSize argument specifies how many hidden
// neurons to use in the non-recurrent output layer.
func NewLSTM(inSize, hiddenSize, hiddenCount, outHiddenSize, outCount int) *Model {
	outNet := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  hiddenSize,
			OutputCount: outHiddenSize,
		},
		&neuralnet.Sigmoid{},
		&neuralnet.DenseLayer{
			InputCount:  outHiddenSize,
			OutputCount: outCount,
		},
	}
	outNet.Randomize()
	outBlock := rnn.NewNetworkBlock(outNet, 0)
	var res rnn.StackedBlock
	for i := 0; i < hiddenCount; i++ {
		if i == 0 {
			res = append(res, rnn.NewLSTM(inSize, hiddenSize))
		} else {
			res = append(res, rnn.NewLSTM(hiddenSize, hiddenSize))
		}
	}
	res = append(res, outBlock)
	return &Model{
		SeqFunc:       &rnn.BlockSeqFunc{Block: res},
		Learner:       res,
		Runner:        &rnn.Runner{Block: res},
		Cost:          &neuralnet.SigmoidCECost{},
		OutActivation: &neuralnet.Sigmoid{},
	}
}
