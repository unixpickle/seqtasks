package main

import (
	"github.com/unixpickle/clockwork"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

// NewCWNN creates a clockwork RNN model.
func NewCWRNN(fullConn bool, inSize, outSize int, freqs []int, stateSizes []int) *Model {
	var block *clockwork.Block
	if fullConn {
		block = clockwork.NewBlockFC(inSize, freqs, stateSizes)
	} else {
		block = clockwork.NewBlock(inSize, freqs, stateSizes)
	}
	stackedBlock := rnn.StackedBlock{
		block,
		rnn.NewNetworkBlock(neuralnet.Network{
			neuralnet.NewDenseLayer(block.OutSize(), outSize),
		}, 0),
	}
	return &Model{
		SeqFunc:       &rnn.BlockSeqFunc{B: stackedBlock},
		Learner:       stackedBlock,
		Runner:        &rnn.Runner{Block: stackedBlock},
		Cost:          &neuralnet.SigmoidCECost{},
		OutActivation: &neuralnet.Sigmoid{},
	}
}
