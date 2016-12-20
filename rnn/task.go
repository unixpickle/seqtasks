package main

import (
	"log"

	"github.com/unixpickle/mnist"
	"github.com/unixpickle/seqtasks"
)

type Task struct {
	Name   string
	Task   seqtasks.Task
	Models map[string]seqtasks.Model

	MaxEpochs    int
	MaxScore     float64
	TrainingSize int
	TestingBatch int
	TestingCount int
}

func (t *Task) Run(modelName string) {
	model, ok := t.Models[modelName]
	if !ok {
		log.Printf("Model \"%s\" not implemented for task \"%s\"", modelName, t.Name)
		return
	}
	log.Printf("Running task \"%s\" with model \"%s\"", t.Name, modelName)
	for i := 0; i < t.MaxEpochs; i++ {
		samples := t.Task.NewSamples(t.TrainingSize)
		model.Train(samples)
		score := t.Task.Score(model, t.TestingBatch, t.TestingCount)
		log.Printf("epoch %d: score=%f", i, score)
		if score >= t.MaxScore {
			break
		}
	}
}

var Tasks = []*Task{
	{
		Name: "XOR last",
		Task: &seqtasks.XORLastTask{SeqLen: 50},
		Models: map[string]seqtasks.Model{
			"lstm":       NewLSTM(1, 40, 1, 40, 1),
			"stack":      NewStructLSTM(Structs["stack"], 1, 40, 1, 40, 1),
			"queue":      NewStructLSTM(Structs["queue"], 1, 40, 1, 40, 1),
			"multistack": NewStructLSTM(Structs["multistack"], 1, 40, 1, 40, 1),
			"multiqueue": NewStructLSTM(Structs["multiqueue"], 1, 40, 1, 40, 1),
			"irnn":       NewIRNN(1, 40, 3, 40, 1, 1),
			"nprnn":      NewNPRNN(1, 40, 3, 40, 1),
			"ffstruct":   NewStructFeedforward(Structs["ffstruct"], 1, 1, 40),
			"hebbnet":    NewHebbNet(1, 20, 2, 40, 1),
			"cwrnn":      NewCWRNN(false, 1, 1, []int{1, 2, 4}, []int{10, 10, 10}),
			"cwrnnfc":    NewCWRNN(true, 1, 1, []int{1, 2, 4}, []int{10, 10, 10}),
		},
		MaxEpochs:    100,
		MaxScore:     1,
		TrainingSize: 100,
		TestingBatch: 10,
		TestingCount: 10,
	},
	{
		Name: "Addition",
		Task: &seqtasks.AdditionTask{MaxDigits: 3, Base: 4},
		Models: map[string]seqtasks.Model{
			"lstm":       NewLSTM(5, 40, 3, 40, 4).UseSoftmax(),
			"stack":      NewStructLSTM(Structs["stack"], 5, 40, 1, 40, 4).UseSoftmax(),
			"queue":      NewStructLSTM(Structs["queue"], 5, 40, 1, 40, 4).UseSoftmax(),
			"multistack": NewStructLSTM(Structs["multistack"], 5, 40, 1, 40, 4).UseSoftmax(),
			"multiqueue": NewStructLSTM(Structs["multiqueue"], 5, 40, 1, 40, 4).UseSoftmax(),
			"irnn":       NewIRNN(5, 40, 3, 40, 4, 1).UseSoftmax(),
			"nprnn":      NewNPRNN(5, 40, 3, 40, 4).UseSoftmax(),
			"ffstruct":   NewStructFeedforward(Structs["ffstruct"], 5, 4, 40).UseSoftmax(),
			"hebbnet":    NewHebbNet(5, 40, 3, 40, 4).UseSoftmax(),
			"cwrnn":      NewCWRNN(false, 5, 4, []int{1, 2, 4, 8}, []int{20, 20, 20, 20}),
			"cwrnnfc":    NewCWRNN(true, 5, 4, []int{1, 2, 4, 8}, []int{20, 20, 20, 20}),
		},
		MaxEpochs:    1000,
		MaxScore:     1,
		TrainingSize: 500,
		TestingBatch: 10,
		TestingCount: 30,
	},
	{
		Name: "Repeat",
		Task: &seqtasks.RepeatTask{
			MinString: 2,
			MaxString: 5,
			MinGap:    0,
			MaxGap:    6,
		},
		Models: map[string]seqtasks.Model{
			"lstm":       NewLSTM(3, 100, 1, 100, 1),
			"stack":      NewStructLSTM(Structs["stack"], 3, 40, 1, 40, 1),
			"queue":      NewStructLSTM(Structs["queue"], 3, 40, 1, 40, 1),
			"multistack": NewStructLSTM(Structs["multistack"], 3, 40, 1, 40, 1),
			"multiqueue": NewStructLSTM(Structs["multiqueue"], 3, 40, 1, 40, 1),
			"irnn":       NewIRNN(3, 100, 1, 100, 1, 0.1),
			"nprnn":      NewNPRNN(3, 40, 1, 40, 1),
			"ffstruct":   NewStructFeedforward(Structs["ffstruct"], 3, 1, 40),
			"hebbnet":    NewHebbNet(3, 20, 2, 40, 1),
			"cwrnn":      NewCWRNN(false, 3, 1, []int{1, 2, 4, 8}, []int{20, 20, 20, 20}),
			"cwrnnfc":    NewCWRNN(true, 3, 1, []int{1, 2, 4, 8}, []int{20, 20, 20, 20}),
		},
		MaxEpochs:    100,
		MaxScore:     1,
		TrainingSize: 300,
		TestingBatch: 10,
		TestingCount: 30,
	},
	{
		Name: "LagEcho",
		Task: &seqtasks.RepeatTask{
			MinString: 1,
			MaxString: 1,
			MinGap:    5,
			MaxGap:    30,
		},
		Models: map[string]seqtasks.Model{
			"lstm":       NewLSTM(3, 20, 1, 20, 1),
			"stack":      NewStructLSTM(Structs["stack"], 3, 40, 1, 40, 1),
			"queue":      NewStructLSTM(Structs["queue"], 3, 40, 1, 40, 1),
			"multistack": NewStructLSTM(Structs["multistack"], 3, 40, 1, 40, 1),
			"multiqueue": NewStructLSTM(Structs["multiqueue"], 3, 40, 1, 40, 1),
			"irnn":       NewIRNN(3, 40, 4, 40, 1, 0.9),
			"nprnn":      NewNPRNN(3, 40, 4, 40, 1),
			"ffstruct":   NewStructFeedforward(Structs["ffstruct"], 3, 1, 40),
			"hebbnet":    NewHebbNet(3, 40, 3, 40, 1),
			"cwrnn":      NewCWRNN(false, 3, 1, []int{1, 2, 4, 8}, []int{20, 20, 20, 20}),
			"cwrnnfc":    NewCWRNN(true, 3, 1, []int{1, 2, 4, 8}, []int{20, 20, 20, 20}),
		},
		MaxEpochs:    1000,
		MaxScore:     1,
		TrainingSize: 300,
		TestingBatch: 10,
		TestingCount: 30,
	},
	{
		Name: "Match Open",
		Task: &seqtasks.MatchOpenTask{
			MinLen:  1,
			MaxLen:  15,
			MaxOpen: 6,
		},
		Models: map[string]seqtasks.Model{
			"lstm":       NewLSTM(3, 40, 1, 40, 1),
			"stack":      NewStructLSTM(Structs["stack"], 3, 40, 1, 40, 1),
			"queue":      NewStructLSTM(Structs["queue"], 3, 40, 1, 40, 1),
			"multistack": NewStructLSTM(Structs["multistack"], 3, 40, 1, 40, 1),
			"multiqueue": NewStructLSTM(Structs["multiqueue"], 3, 40, 1, 40, 1),
			"irnn":       NewIRNN(3, 40, 3, 40, 1, 1),
			"nprnn":      NewNPRNN(3, 40, 3, 40, 1),
			"ffstruct":   NewStructFeedforward(Structs["ffstruct"], 3, 1, 40),
			"hebbnet":    NewHebbNet(3, 20, 2, 40, 1),
			"cwrnn":      NewCWRNN(false, 3, 1, []int{1, 2, 4, 8}, []int{20, 20, 20, 20}),
			"cwrnnfc":    NewCWRNN(true, 3, 1, []int{1, 2, 4, 8}, []int{20, 20, 20, 20}),
		},
		MaxEpochs:    50,
		MaxScore:     1,
		TrainingSize: 100,
		TestingBatch: 10,
		TestingCount: 30,
	},
	{
		Name: "Random Recall",
		Task: &seqtasks.RandomRecallTask{Bits: 4, SeqLen: 30},
		Models: map[string]seqtasks.Model{
			"lstm":       NewLSTM(6, 40, 1, 40, 4),
			"stack":      NewStructLSTM(Structs["stack"], 6, 40, 1, 40, 4),
			"queue":      NewStructLSTM(Structs["queue"], 6, 40, 1, 40, 4),
			"multistack": NewStructLSTM(Structs["multistack"], 6, 40, 1, 40, 4),
			"multiqueue": NewStructLSTM(Structs["multiqueue"], 6, 40, 1, 40, 4),
			"irnn":       NewIRNN(6, 40, 3, 40, 4, 1),
			"nprnn":      NewNPRNN(6, 40, 2, 40, 4),
			"ffstruct":   NewStructFeedforward(Structs["ffstruct"], 6, 4, 40),
			"hebbnet":    NewHebbNet(6, 20, 2, 40, 4),
			"cwrnn":      NewCWRNN(false, 6, 4, []int{1, 2, 4, 8}, []int{20, 20, 20, 20}),
			"cwrnnfc":    NewCWRNN(true, 6, 4, []int{1, 2, 4, 8}, []int{20, 20, 20, 20}),
		},
		MaxEpochs:    1000,
		MaxScore:     1,
		TrainingSize: 1000,
		TestingBatch: 10,
		TestingCount: 100,
	},
	{
		Name: "Match Multi",
		Task: &seqtasks.MatchMultiTask{
			TypeCount: 4,
			MinLen:    1,
			MaxLen:    8,
			CloseProb: 0.3,
		},
		Models: map[string]seqtasks.Model{
			"lstm":       NewLSTM(4*2+1, 100, 2, 100, 4+1),
			"stack":      NewStructLSTM(Structs["stack"], 4*2+1, 40, 1, 40, 4+1),
			"queue":      NewStructLSTM(Structs["queue"], 4*2+1, 40, 1, 40, 4+1),
			"multistack": NewStructLSTM(Structs["multistack"], 4*2+1, 40, 1, 40, 4+1),
			"multiqueue": NewStructLSTM(Structs["multiqueue"], 4*2+1, 40, 1, 40, 4+1),
			"irnn":       NewIRNN(4*2+1, 40, 3, 40, 4+1, 1),
			"nprnn":      NewNPRNN(4*2+1, 40, 3, 40, 4+1),
			"ffstruct":   NewStructFeedforward(Structs["ffstruct"], 4*2+1, 4+1, 40),
			"hebbnet":    NewHebbNet(4*2+1, 20, 2, 40, 4+1),
			"cwrnn":      NewCWRNN(false, 4*2+1, 4+1, []int{1, 2, 4, 8}, []int{20, 20, 20, 20}),
			"cwrnnfc":    NewCWRNN(true, 4*2+1, 4+1, []int{1, 2, 4, 8}, []int{20, 20, 20, 20}),
		},
		MaxEpochs:    1000,
		MaxScore:     1,
		TrainingSize: 3000,
		TestingBatch: 20,
		TestingCount: 100,
	},
	{
		Name: "MNIST",
		Task: &seqtasks.MNISTTask{
			Training: mnist.LoadTrainingDataSet(),
			Testing:  mnist.LoadTestingDataSet(),
		},
		Models: map[string]seqtasks.Model{
			"lstm":       NewLSTM(2, 100, 2, 100, 10).UseSoftmax(),
			"stack":      NewStructLSTM(Structs["stack"], 2, 40, 1, 40, 10).UseSoftmax(),
			"queue":      NewStructLSTM(Structs["queue"], 2, 40, 1, 40, 10).UseSoftmax(),
			"multistack": NewStructLSTM(Structs["multistack"], 2, 40, 1, 40, 10).UseSoftmax(),
			"multiqueue": NewStructLSTM(Structs["multiqueue"], 2, 40, 1, 40, 10).UseSoftmax(),
			"irnn":       NewIRNN(2, 40, 3, 40, 10, 1).UseSoftmax(),
			"nprnn":      NewNPRNN(2, 40, 3, 40, 10).UseSoftmax(),
			"ffstruct":   NewStructFeedforward(Structs["ffstruct"], 2, 10, 40).UseSoftmax(),
			"hebbnet":    NewHebbNet(2, 20, 2, 40, 10).UseSoftmax(),
			"cwrnn": NewCWRNN(false, 2, 10, []int{1, 2, 4, 8, 16, 32, 64, 128, 256, 512},
				[]int{20, 20, 20, 20, 20, 20, 20, 20, 20, 20}),
			"cwrnnfc": NewCWRNN(true, 2, 10, []int{1, 2, 4, 8, 16, 32, 64, 128, 256, 512},
				[]int{20, 20, 20, 20, 20, 20, 20, 20, 20, 20}),
		},
		MaxEpochs:    10000,
		MaxScore:     1,
		TrainingSize: 100,
		TestingBatch: 1,
		TestingCount: 100,
	},
}
