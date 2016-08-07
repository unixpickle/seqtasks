package seqtasks

import (
	"math/rand"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

// MatchOpenTask is a task equivalent to matching opening
// parantheses with closing parentheses.
// Sequences include a bunch of open and close "parentheses"
// followed by a "close all" indicator to request that the
// model close all the unclosed parentheses.
type MatchOpenTask struct {
	// MinLen is the minimum number of parentheses to
	// give the model before requesting a "close all".
	MinLen int

	// MaxLen is the maximum number of parentheses to
	// give the model before requesting a "close all".
	MaxLen int

	// MaxOpen is the maximum number of unclosed
	// parentheses to generate.
	MaxOpen int
}

// InputSize returns 3, since the alphabet includes an
// open parentheses, a close parentheses, and a "close all"
// indicator.
func (m *MatchOpenTask) InputSize() int {
	return 3
}

// OutputSize returns 1, since the model is only required
// to output 1s for a certain number of times before giving
// a 0 to indicate all parentheses are closed.
func (m *MatchOpenTask) OutputSize() int {
	return 1
}

// NewSamples generates sample vectors.
func (m *MatchOpenTask) NewSamples(n int) sgd.SampleSet {
	var res sgd.SliceSampleSet
	for i := 0; i < n; i++ {
		var sample seqtoseq.Sample
		stringSize := rand.Intn(m.MaxLen-m.MinLen+1) + m.MinLen

		stringCloses := make([]bool, stringSize)
		minClose := stringSize - m.MaxOpen
		if minClose < 0 {
			minClose = 0
		}
		closeCount := rand.Intn(stringSize-minClose) + minClose
		perm := rand.Perm(stringSize)
		for j := 0; j < closeCount; j++ {
			stringCloses[perm[j]] = true
		}

		for _, close := range stringCloses {
			if close {
				sample.Inputs = append(sample.Inputs, []float64{0, 1, 0})
			} else {
				sample.Inputs = append(sample.Inputs, []float64{1, 0, 0})
			}
			sample.Outputs = append(sample.Outputs, []float64{0})
		}
		sample.Inputs = append(sample.Inputs, []float64{0, 0, 1})
		sample.Outputs = append(sample.Outputs, []float64{0})
		for j := 0; j < stringSize-closeCount; j++ {
			sample.Inputs = append(sample.Inputs, []float64{0, 0, 0})
			sample.Outputs = append(sample.Outputs, []float64{1})
		}
		sample.Inputs = append(sample.Inputs, []float64{0, 0, 0})
		sample.Outputs = append(sample.Outputs, []float64{0})
		res = append(res, sample)
	}
	return res
}

// Score computes the fraction of correct (rounded) outputs
// after "close all" symbols.
func (m *MatchOpenTask) Score(model Model, batchSize, batchCount int) float64 {
	return roundedBinaryTailScore(m, model, batchSize, batchCount, func(s []linalg.Vector) int {
		for i, x := range s {
			if x[2] == 1 {
				return i + 1
			}
		}
		panic("no tail found")
	})
}
