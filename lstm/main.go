package main

import (
	"log"

	"github.com/unixpickle/seqtasks"
	"github.com/unixpickle/weakai/neuralnet"
)

type Task struct {
	Name  string
	Task  seqtasks.Task
	Model seqtasks.Model

	MaxEpochs    int
	MaxScore     float64
	TrainingSize int
	TestingBatch int
	TestingCount int
}

func main() {
	tasks := []Task{
		{
			Name: "XOR last",
			Task: &seqtasks.XORLastTask{SeqLen: 50},
			Model: &BlockModel{
				Block:         NewBlock(1, 40, 40, 1),
				Cost:          &neuralnet.SigmoidCECost{},
				OutActivation: &neuralnet.Sigmoid{},
			},
			MaxEpochs:    50,
			MaxScore:     1,
			TrainingSize: 100,
			TestingBatch: 10,
			TestingCount: 10,
		},
		{
			Name: "Repeat",
			Task: &seqtasks.RepeatTask{
				MinString: 2,
				MaxString: 5,
				MinGap:    0,
				MaxGap:    6,
			},
			Model: &BlockModel{
				Block:         NewBlock(3, 100, 100, 1),
				Cost:          &neuralnet.SigmoidCECost{},
				OutActivation: &neuralnet.Sigmoid{},
			},
			MaxEpochs:    100,
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
			Model: &BlockModel{
				Block:         NewBlock(3, 40, 40, 1),
				Cost:          &neuralnet.SigmoidCECost{},
				OutActivation: &neuralnet.Sigmoid{},
			},
			MaxEpochs:    50,
			MaxScore:     1,
			TrainingSize: 100,
			TestingBatch: 10,
			TestingCount: 30,
		},
		{
			Name: "Match Multi",
			Task: &seqtasks.MatchMultiTask{
				TypeCount: 4,
				MinLen:    1,
				MaxLen:    5,
				CloseProb: 0.3,
			},
			Model: &BlockModel{
				Block:         NewDeepBlock(4*2+1, 100, 2, 100, 4+1),
				Cost:          &neuralnet.SigmoidCECost{},
				OutActivation: &neuralnet.Sigmoid{},
			},
			MaxEpochs:    50,
			MaxScore:     1,
			TrainingSize: 400,
			TestingBatch: 10,
			TestingCount: 30,
		},
	}
	for _, task := range tasks {
		log.Println("Running task", task.Name, "...")
		samples := task.Task.NewSamples(task.TrainingSize)
		for i := 0; i < task.MaxEpochs; i++ {
			task.Model.Train(samples)
			score := task.Task.Score(task.Model, task.TestingBatch, task.TestingCount)
			log.Printf("epoch %d: score=%f", i, score)
			if score >= task.MaxScore {
				break
			}
		}
	}
}
