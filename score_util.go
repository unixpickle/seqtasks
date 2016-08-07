package seqtasks

import (
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

// roundedBinaryScore returns the fraction of outputs
// match the correct outputs when rounded to 0 or 1.
func roundedBinaryScore(t Task, m Model, batchSize, batchCount int) float64 {
	return roundedBinaryTailScore(t, m, batchSize, batchSize, func(s []linalg.Vector) int {
		return 0
	})
}

// roundedBinaryTailScore is like roundedBinaryScore, but it
// only counts outputs in the "tail" of each sequence.
// The tailFunc argument takes input sequences and returns
// the start index of the tail for that sequence.
func roundedBinaryTailScore(t Task, m Model, batchSize, batchCount int,
	tailFunc func(seq []linalg.Vector) int) float64 {
	var totalOutputs int
        var totalCorrect int
        for i := 0; i < batchCount; i++ {
                batch := t.NewSamples(batchSize)
                var inputs [][]linalg.Vector
                var expected [][]linalg.Vector
                for i := 0; i < batch.Len(); i++ {
                        sample := batch.GetSample(i).(seqtoseq.Sample)
                        inputs = append(inputs, sample.Inputs)
                        expected = append(expected, sample.Outputs)
                }
                actual := m.Run(inputs)
                for lane, expSeq := range expected {
			tailIdx := tailFunc(inputs[lane])
                        actSeq := actual[lane][tailIdx:]
                        for t, expVec := range expSeq[tailIdx:] {
                                actVec := actSeq[t]
                                for j, x := range expVec {
                                        a := int(actVec[j] + 0.5)
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

