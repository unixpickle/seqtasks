package main

import (
	"fmt"
	"log"

	"github.com/unixpickle/seqtasks"
	"github.com/unixpickle/weakai/neuralnet"
)

type Task struct {
	Name  string
	Task  seqtasks.Task
	Model seqtasks.Model
}

func main() {
	tasks := []Task{
		{
			Name: "XOR last",
			Task: &seqtasks.XORLastTask{SeqLen: 50},
			Model: &BlockModel{
				Block:         NewBlock(1, 40, 40),
				Cost:          &neuralnet.SigmoidCECost{},
				OutActivation: &neuralnet.Sigmoid{},
			},
		},
	}
	for _, task := range tasks {
		log.Println("Running task", task.Name, "...")
		samples := task.Task.NewSamples(100)
		for {
			task.Model.Train(samples)
			fmt.Println("ran iteration")
		}
	}
}
