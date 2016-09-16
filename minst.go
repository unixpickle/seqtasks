package seqtasks

import (
	"math/rand"

	"github.com/unixpickle/mnist"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

// MNISTTask requires the model to classify handwritten
// digits, given a string of pixels comprising an image.
type MNISTTask struct {
	Training mnist.DataSet
	Testing  mnist.DataSet
}

// InputSize returns 2, since there is a pixel input and
// an "end-of-digit" input.
func (m *MNISTTask) InputSize() int {
	return 2
}

// OutputSize returns 10, since the model needs to be
// able to choose between 10 digit classes.
func (m *MNISTTask) OutputSize() int {
	return 10
}

// NewSamples creates a list of training sample sequences.
func (m *MNISTTask) NewSamples(n int) sgd.SampleSet {
	var res sgd.SliceSampleSet
	for i := 0; i < n; i++ {
		var resSample seqtoseq.Sample
		sample := m.Training.Samples[rand.Intn(len(m.Training.Samples))]
		for _, x := range sample.Intensities {
			resSample.Inputs = append(resSample.Inputs, []float64{x, 0})
			resSample.Outputs = append(resSample.Outputs, make(linalg.Vector, 10))
		}
		resSample.Inputs = append(resSample.Inputs, []float64{0, 1})
		outVec := make(linalg.Vector, 10)
		outVec[sample.Label] = 1
		resSample.Outputs = append(resSample.Outputs, outVec)
		res = append(res, resSample)
	}
	return res
}

// Score computes the fraction of correctly classified
// digits, as measured by the testing data set.
func (m *MNISTTask) Score(model Model, batchSize, batchCount int) float64 {
	var correct int
	for i := 0; i < batchCount; i++ {
		var labels []int
		var sequences [][]linalg.Vector
		for j := 0; j < batchSize; j++ {
			sample := m.Testing.Samples[rand.Intn(len(m.Testing.Samples))]
			labels = append(labels, sample.Label)
			var in []linalg.Vector
			for _, x := range sample.Intensities {
				in = append(in, []float64{x, 0})
			}
			in = append(in, []float64{0, 1})
			sequences = append(sequences, in)
		}
		outs := model.Run(sequences)
		for j, label := range labels {
			lastOut := outs[j][len(outs[j])-1]
			if maxIdx(lastOut) == label {
				correct++
			}
		}
	}
	return float64(correct) / float64(batchSize*batchCount)
}

func maxIdx(v linalg.Vector) int {
	maxVal := v[0]
	maxIdx := 0
	for i, x := range v {
		if x > maxVal {
			maxIdx = i
			maxVal = x
		}
	}
	return maxIdx
}
