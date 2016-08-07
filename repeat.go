package seqtasks

import (
	"math/rand"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

// RepeatTask spits out a string of bits, then a bunch of zeroes,
// and then requires the model to output the original string.
type RepeatTask struct {
	// MinString is the minimum length for the string.
	MinString int

	// MaxString is the maximum length for the string.
	MaxString int

	// MinGap is the minimum number of zeroes between giving
	// the string and requesting it back.
	MinGap int

	// MaxGap is the maximum number of zeroes between giving
	// the string and requesting it back.
	MaxGap int
}

// InputSize returns 3, since the first input is for data, the
// second to indicate the end of string, and the third to
// request the original string.
func (r *RepeatTask) InputSize() int {
	return 3
}

// OutputSize returns 1, since the model only needs to output
// a single bit at a time.
func (r *RepeatTask) OutputSize() int {
	return 1
}

// NewSamples creates a list of sample sequences.
func (r *RepeatTask) NewSamples(n int) sgd.SampleSet {
	var res sgd.SliceSampleSet
	for i := 0; i < n; i++ {
		var sample seqtoseq.Sample
		stringLen := rand.Intn(r.MaxString-r.MinString+1) + r.MinString
		gapLen := rand.Intn(r.MaxGap-r.MinGap+1) + r.MinGap
		for j := 0; j < stringLen; j++ {
			val := float64(rand.Intn(2))
			sample.Inputs = append(sample.Inputs, []float64{val, 0, 0})
			sample.Outputs = append(sample.Outputs, []float64{0})
		}
		sample.Inputs = append(sample.Inputs, []float64{0, 1, 0})
		sample.Outputs = append(sample.Outputs, []float64{0})
		for j := 0; j < gapLen; j++ {
			sample.Inputs = append(sample.Inputs, []float64{0, 0, 0})
			sample.Outputs = append(sample.Outputs, []float64{0})
		}
		sample.Inputs = append(sample.Inputs, []float64{0, 0, 1})
		sample.Outputs = append(sample.Outputs, []float64{0})
		for j := 0; j < stringLen; j++ {
			out := sample.Inputs[j][0]
			sample.Inputs = append(sample.Inputs, []float64{0, 0, 0})
			sample.Outputs = append(sample.Outputs, []float64{out})
		}
		res = append(res, sample)
	}
	return res
}

// Score computes the fraction of outputs the model gets
// correct, not including all of the zero outputs leading
// up to the recall phase.
// Output values from the model are rounded to 0 or 1.
func (r *RepeatTask) Score(m Model, batchSize, batchCount int) float64 {
	var totalOutputs int
	var totalCorrect int
	for i := 0; i < batchCount; i++ {
		batch := r.NewSamples(batchSize)
		var inSeqs [][]linalg.Vector
		var expected [][]linalg.Vector
		for j := 0; j < batch.Len(); j++ {
			sample := batch.GetSample(j).(seqtoseq.Sample)
			inSeqs = append(inSeqs, sample.Inputs)
			expected = append(expected, sample.Outputs)
		}
		actual := m.Run(inSeqs)
		for lane, expSeq := range expected {
			actSeq := actual[lane]
			var recallPhase bool
			for j, expVec := range expSeq {
				if !recallPhase {
					inVal := inSeqs[lane][j][2]
					recallPhase = (inVal == 1)
					continue
				}
				actVec := actSeq[j]
				for k, x := range expVec {
					a := int(actVec[k] + 0.5)
					if a < 0 {
						a = 0
					} else if a > 1 {
						a = 1
					}
					if float64(a) == x {
						totalCorrect++
					}
					totalOutputs++
				}
			}
		}
	}
	return float64(totalCorrect) / float64(totalOutputs)
}
