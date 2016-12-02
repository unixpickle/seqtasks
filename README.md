# seqtasks

This project provides a bunch of synthetic tests and benchmarks for recurrent architectures. Using these benchmarks, it should be straightforward to compare new recurrent architectures to existing ones such as LSTM or GRU.

The [rnn](rnn) directory is a program which runs a bunch of standard benchmarks on a bunch of standard (and some non-standard) recurrent architectures. The [stochnet](stochnet) directory runs similar benchmarks, but on a different, experimental model of mine.

More information can be found in the [Godoc](https://godoc.org/github.com/unixpickle/seqtasks).
