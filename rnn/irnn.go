package main

import (
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

// NewIRNN creates an identity RNN model.
// The arguments are similar to NewLSTM.
// There is an extra argument for the IRNN identity
// scale hyperparameter.
func NewIRNN(inSize, hiddenSize, hiddenCount, outHiddenSize, outCount int,
	idScale float64) *Model {
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
			res = append(res, rnn.NewIRNN(inSize, hiddenSize, idScale))
		} else {
			res = append(res, rnn.NewIRNN(hiddenSize, hiddenSize, idScale))
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
