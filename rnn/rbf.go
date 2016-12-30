package main

import (
	"math"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rbf"
	"github.com/unixpickle/weakai/rnn"
)

// NewRBF creates a recurrent RBF network.
func NewRBF(inSize, hiddenSize, hiddenCount, outHiddenSize, outCount int) *Model {
	return NewBlockModel(inSize, hiddenSize, hiddenCount, outHiddenSize,
		outCount, func(i, h int) rnn.Block {
			net := &rbf.Network{
				DistLayer:  rbf.NewDistLayer(i+h, h, 1),
				ScaleLayer: rbf.NewScaleLayer(h, 1/math.Sqrt(float64(i+h))),
				ExpLayer:   &rbf.ExpLayer{Normalize: true},
				OutLayer:   neuralnet.NewDenseLayer(h, h),
			}
			net.OutLayer.Weights.Data.Vector = linalg.RandVector(h * h)
			return &rnn.StateOutBlock{
				Block: rnn.NewNetworkBlock(neuralnet.Network{net}, h),
			}
		})
}
