# Cserve Lightweight Benchmarker
This benchmark operates entirely external to any serving framework, and can easily be extended and modified. Provides a variety of statistics and profiling modes. It is intended to be a standalone tool for precise statistically significant benchmarking with a particular input/output distribution. Each request consists of a single prompt and single decode. 

This benchmark basically sends out as many requests as you specify, with the length of the request and time that request hits the model server based on distributions that you specify.

### Installation
1) Install rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` and restart the shell. See: https://www.rust-lang.org/tools/install
2) Run `cargo build` in `benchmarks/lightweight_benchmark`

### Usage
1) Ensure the framework you wish to use is started (CServe or vLLM) so that the generate API is exposed.
2) Find the binary under target/lightweight_benchmarker. Then launch it with all the arguments specified below.

### Arguments and Feature List
* `--help`: Find all of the options below.
* `--verbose`: Enables verbose printing of various INFOs.
* `--output-file`: Specify a filename for the output json. Otherwise, prints to stdout.
* `--benchmark-note`: Spcify a note (string) to include in the output json. Default: empty.
* `--config-file`: This is an alternative way of passing in all the arguments below. They are specified in camelCase (e.g. `--request-rate` becomes `requestRate`). If both a config file and a cli argument is specified for the same parameter, the cli argument takes precedence. This makes it possible to use a config file to specify your own defaults.
* `--text-file`: Specifies a text file to use as prompt input. This text file should consist of a single line. This line should contain as many tokens as necessary for your requests. Useful for speculative decoding type tasks.
* `--tokenizer-name`: Huggingface name of the model you intend to use. This helps tokenize an exact number of tokens for the prompt.
* `--hostname`: Specify the hostname where the endpoint is located. Default: localhost 
* `--port`: Specify the port on hostname where the endpoint is located. Default: 8080
* `--framework`: Specify the framework. Can be one of:
    * vllm
    * cserve
* `--request-distribution`: Specify the distribution for input requests to arrive. This controls how fast requests reach the inference endpoint. Can be one of:
    * poisson (with $request\_rate$).
    * even (Non-random where requests arrive every $1/request \textunderscore rate$ seconds). That is, $request \textunderscore rate$ number of requests reach the inference endpoint every second, evenly spaced.
    * same (Start at 0). This means the benchmarker sends all requests at the exact same time right at the beginning. Default.
* `--num-requests`: Number of requests to launch per trial. Default: 1.
* `--num-samples`: Number of times to run the experiment. This basically is the number of trials to increase statistical confidence in the benchmark results. Default: 1.
* `--use-beam-search`: Whether to use beam search or not. Default: False.
* `--best-of`: Beam width, when beam search is on. Otherwise, number of output sequences produced from prompt. Default: 1.
* `--request-rate`: Request rate used in the request distribution. Default: 1.0
* `--prompt-low`: The (inclusive) start of the range of the uniform distribution from which prompt lengths are sampled from.
* `--prompt-high`: The (exclusive) end of the range of the uniform distribution from which prompt lengths are sampled from.
* `--decode-low`: The (inclusive) start of the range of the uniform distribution from which decode lengths are sampled from.
* `--decode-high`: The (exclusive) end of the range of the uniform distribution from which decode lengths are sampled from.

### Output Parameters
* `n`: This is the total number of experiments run. The exact same as `--num-samples` above.
* `total_time`: The total time for all requests to complete *averaged* over `n`.
* `throughput`: Number of requests processed per second.
* `avg_latency`: The average time for one request to complete end-to-end, that is between sending the request out and receiving the response with all output tokens in full.
* `avg_latency_per_token`: The average time for one token to complete. This is computed by taking each request time averaged over the number of prompt tokens + number of output tokens, then averaging this over all requests.
* `avg_latency_per_output_token`: The average time for one output token to complete. This is the computed by taking each request time and averaging over the number of output tokens, then averaging over all requests.
* `--note`: The exact note specified by `--benchmark-note` in the input. Useful to annotate or comment on experiments and have it noted in the output.

### Examples
Consider a situation where we run our benchmarker with `--num-requests` as 2 and `--num-samples` as 1. We might then end up with the following scenario where \* represents time a request spends in the inference engine, and \- represents time when the request is not in the engine.

```
R1 --********************************----
R2 -------------****************---------
     1          2              3    4
```

We annotate interesting points to note.


At point 1, R1 starts, while at point 2, R2 starts, i.e. is sent from the benchmarker. This will depend entirely on the `--request-rate` and `--request-distribution` that we have set. Point 3 represents when R2 has finished and returned its output completely.

Point 4 represents when R1 has finished and returned its output completely. The `total_time` that is returned here will be time 4 while the throughput will be time 4 over the number of requests.

We are also interested in the latencies of each request, which is what the other 3 output statistics in **Output Parameters** depend on. This is time 3 less time 2, and time 4 less time 1. This latency will depend on how the inference engine we are trying to benchmark handles requests, how performant it is, as well as the `--prompt-` and `--decode-` parameters that we specify in the input. It may also depend on the `--text-file` parameter.

If we set `--num-samples` to a number other than 1, we will run the above experiment that number of times. All distributions will be sampled independently between statistics. This gives us greater confidence in our results. This is to say, if `--num-samples` is 3, we will send R1 and R2 out for a first time, gather statistics, send R1 and R2 out for a second time (the times 1, 2, 3, 4 could be different this time due to the random distributions parametrized by `--request-distribution`, `--prompt-`, `--decode-` being sampled again, as well as random effects) gather statistics, and send R1 and R2 a third time, and gather statistics. Then, all statstics will be *averaged* over `--num-samples` in the output. This is why `n` is output as well to note that the other statistics were gathered by an average over these number of trials.
