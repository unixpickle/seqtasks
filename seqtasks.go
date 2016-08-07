// Package seqtasks provides a number of tests and benchmarks
// for ML architectures that map sequences to sequences.
package seqtasks

import (
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

// A Task is a benchmark or test for a sequence-to-sequence
// model to learn.
type Task interface {
	// InputSize is the size of the input vectors at each
	// timestep.
	InputSize() int

	// OutputSize is the size of the output vectors at each
	// timestep.
	OutputSize() int

	// NewSamples creates a new set of training or testing
	// samples for this task.
	NewSamples(count int) seqtoseq.Sample
}

// A Model learns to solve Tasks.
type Model interface {
	// Train performs a round of training for some samples.
	Train(samples sgd.SampleSet)

	// Run runs a slice of sequences on the model and returns
	// the output vectors.
	Run(inputs [][]linalg.Vector)
}
