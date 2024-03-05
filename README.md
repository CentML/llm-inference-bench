# Cserve Lightweight Benchmarker
This benchmark operates entirely external to any serving framework, and can easily be extended and modified. Provides a variety of statistics and profiling modes. It is intended to be a standalone tool for precise statistically significant benchmarking with a particular input/output distribution. Each request consists of a single prompt and single decode. 

### Installation
1) Install rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` and restart the shell. See: https://www.rust-lang.org/tools/install
2) Run `cargo build` in `benchmarks/lightweight_benchmark`

### Usage
1) Ensure the framework you wish to use is started (CServe or vLLM) so that the generate API is exposed.
2) Find the binary under target/lightweight_benchmarker. Then launch it with all the arguments specified below.

### Arguments and Feature List
* `--help`: Find all of the options below.
* `--verbose`: Enables verbose printing of various INFOs.
* `--output-file`: Specify a filename for the output json. Otherwise, prints to the standard out.
* `--benchmark-note`: Spcify a note (string) to include in the output json. Default: empty.
* `--config-file`: This is an alternative way of passing in all the arguments below. They are specified in camelCase (e.g. `--request-rate` becomes `requestRate`). If both a config file and a cli argument is specified for the same parameter, the cli argument takes precedence. This makes it possible to use a config file to specify your own defaults.
* `--text-file`: Specifies a text file to use as prompt input. Useful for speculative decoding type tasks.
* `--tokenizer-name`: Name of the model you intend to use. This helps tokenize an exact number of tokens for the prompt.
* `--hostname`: Specify the hostname where the endpoint is located. Default: localhost 
* `--port`: Specify the port on hostname where the endpoint is located. Default: 8080
* `--framework`: Specify the framework. Can be one of:
    * vllm
    * cserve
* `--request-distribution`: Specify the distribution for input requests to arrive. Can be one of:
    * poisson (with $request_rate$)
    * even (Non-random wherecrequests arrive every $1/request_rate$)
    * same (Start at 0). Default.
* `--num-samples`: Number of times to run the experiment. Default: 1.
* `--num-requests`: Number of requests to launch per trial. Default: 1.
* `--use-beam-search`: Whether to use beam search or not. Default: False.
* `--best-of`: Beam width, when beam search is on. Otherwise, number of output sequences produced from prompt. Default: 1.
* `--request-rate`: Request rate used in the request distribution. Default: 1.0
* `--prompt-low`: The (inclusive) start of the range of the uniform distribution from which prompt lengths are sampled from.
* `--prompt-high`: The (exclusive) end of the range of the uniform distribution from which prompt lengths are sampled from.
* `--decode-low`: The (inclusive) start of the range of the uniform distribution from which decode lengths are sampled from.
* `--decode-high`: The (exclusive) end of the range of the uniform distribution from which decode lengths are sampled from.
