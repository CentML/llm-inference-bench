use clap::Parser;
use colored::Colorize;
use core::ops::Range;
use futures::future::join_all;
use rand_distr::{Distribution, Exp, Uniform};
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use tokenizers::tokenizer::Tokenizer;
use tokio::time::{sleep, Duration};

#[derive(Parser, Debug, Deserialize, Default)]
struct Args {
    #[serde(rename = "textFile", default)]
    #[arg(long, required(false))]
    text_file: Option<String>,
    #[serde(rename = "tokenizerName", default)]
    #[arg(long, required(false))]
    tokenizer_name: Option<String>,
    #[serde(rename = "requestDistribution", default)]
    #[arg(long, required(false))]
    request_distribution: Option<String>,
    #[serde(rename = "numRequests", default)]
    #[arg(long, required(false))]
    num_requests: Option<usize>,
    #[serde(rename = "requestRate", default)]
    #[arg(long, required(false))]
    request_rate: Option<f32>,
    #[serde(rename = "promptLow", default)]
    #[arg(long, required(false))]
    prompt_low: Option<u32>,
    #[serde(rename = "promptHigh", default)]
    #[arg(long, required(false))]
    prompt_high: Option<u32>,
    #[serde(rename = "decodeLow", default)]
    #[arg(long, required(false))]
    decode_low: Option<u32>,
    #[serde(rename = "decodeHigh", default)]
    #[arg(long, required(false))]
    decode_high: Option<u32>,
    #[serde(rename = "port", default)]
    #[arg(long, required(false))]
    port: Option<u32>,
    #[serde(rename = "bestOf", default)]
    #[arg(long, required(false))]
    best_of: Option<u32>,
    #[serde(rename = "framework", default)]
    #[arg(long, required(false))]
    framework: Option<String>,
    #[serde(rename = "hostname", default)]
    #[arg(long, required(false))]
    hostname: Option<String>,
    #[serde(rename = "useBeamSearch", default)]
    #[arg(long, required(false))]
    use_beam_search: Option<bool>,
    #[serde(rename = "numSamples", default)]
    #[arg(long, required(false))]
    num_samples: Option<usize>,
    #[serde(skip)]
    #[arg(long, required(false))]
    benchmark_note: Option<String>,
    #[serde(skip)]
    #[arg(long, required(false))]
    output_file: Option<String>,
    #[serde(skip)]
    #[arg(long, required(false))]
    verbose: bool,
    #[serde(skip)]
    #[arg(long, required(false))]
    config_file: Option<String>,
}

#[derive(Debug)]
enum InputType {
    File(String),
    Random(u32),
}

#[derive(Debug)]
enum RequestDistribution {
    Poisson(f32),
    Even(f32),
    Same(),
}

// Framework request types. New frameworks can be added here.
enum Framework {
    Cserve(CserveBody),
    Vllm(VllmBody),
}
#[derive(Serialize, Deserialize, Debug)]
struct SamplingParams {
    n: u32,
    max_tokens: u32,
    best_of: u32,
    use_beam_search: bool,
    temperature: f32,
    top_p: f32,
    ignore_eos: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct CserveBody {
    prompt: String,
    sampling_params: SamplingParams,
    stream: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct VllmBody {
    prompt: String,
    stream: bool,
    n: u32,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    ignore_eos: bool,
    use_beam_search: bool,
    best_of: u32,
}
#[derive(Debug)]
struct ResponseStatistics {
    latency: Duration,
    latency_per_token: Duration,
    latency_per_output_token: Duration,
    success: bool,
}

#[derive(Debug, Serialize)]
struct BenchmarkResults {
    n: usize,
    total_time: f32,
    throughput: f32,
    avg_latency: f32,
    avg_latency_per_token: f32,
    avg_latency_per_output_token: f32,
    note: String,
}

/// Encodes a string, truncates the encoding to the desired length, and decodes it back to a string
fn tokenize(
    tokenizer: &Tokenizer,
    input_type: &InputType,
    len: usize,
    verbose: bool,
) -> (String, usize) {
    let num_trials: usize = 10; // Number of times to try to get a valid encoding
    let encoding: String = match input_type {
        InputType::File(s) => {
            let mut target_len: usize = len;
            let mut return_string: String = String::new();
            for trial in 0..num_trials {
                let encoding: tokenizers::Encoding = tokenizer
                    .encode(s.as_str(), false)
                    .expect("Failed to encode");
                return_string = tokenizer
                    .decode(&encoding.get_ids()[0..target_len], false)
                    .expect("Failed to decode");
                let check_len = tokenizer
                    .encode(return_string.as_str(), false)
                    .expect("Failed to encode")
                    .get_ids()
                    .len();
                if verbose {
                    println!(
                        "{} Trial: {} Target Prompt Length: {}, Achieved Prompt Length: {}",
                        "INFO".blue().bold(),
                        trial,
                        len.to_string().green().bold(),
                        check_len.to_string().green().bold()
                    );
                }
                if check_len == len {
                    break;
                } else if check_len > len {
                    target_len -= 2; // Need to assymetrically remove tokens to avoid getting stuck.
                } else {
                    target_len += 1;
                }
            }
            return_string
        }
        InputType::Random(vocab_size) => {
            let mut ids: Vec<u32> = Vec::new();
            for _ in 0..len {
                ids.push(rand::random::<u32>() % vocab_size);
            }
            let mut return_string: String = String::new();
            for trial in 0..num_trials {
                return_string = tokenizer
                    .decode(&ids, false)
                    .expect("Failed to decode random input");
                let check_len = tokenizer
                    .encode(return_string.as_str(), false)
                    .expect("Failed to encode")
                    .get_ids()
                    .len();
                if verbose {
                    println!(
                        "{} Trial: {} Target Prompt Length: {}, Achieved Prompt Length: {}",
                        "INFO".blue().bold(),
                        trial,
                        len.to_string().green().bold(),
                        check_len.to_string().green().bold()
                    );
                }
                if check_len == len {
                    break;
                } else if check_len > len {
                    ids.pop();
                    ids.pop();
                } else {
                    ids.push(rand::random::<u32>() % vocab_size);
                }
            }
            return_string
        }
    };
    let achieved_len: usize = tokenizer
        .encode(encoding.as_str(), false)
        .expect("Failed to encode")
        .get_ids()
        .len();
    if len != achieved_len {
        println!(
            "{}: Failed to get an encoding of the correct length. Expected {}, got {}",
            "WARNING".yellow().bold(),
            len,
            achieved_len
        );
    }
    (encoding, achieved_len)
}

/// Generates a vector of delays between requests with a Poisson distribution
fn get_delays(num_requests: usize, request_distribution: &RequestDistribution) -> Vec<f32> {
    fn get_poisson_delays(request_rate: f32, num_requests: usize) -> Vec<f32> {
        let exp: Exp<f32> = Exp::new(request_rate).unwrap();
        let mut v: Vec<f32> = Vec::new();
        v.push(0.0);
        for _i in 1..num_requests {
            let delay: f32 = exp.sample(&mut rand::thread_rng()) + v.last().unwrap();
            v.push(delay);
        }
        v
    }

    fn get_even_delays(request_rate: f32, num_requests: usize) -> Vec<f32> {
        let mut v: Vec<f32> = Vec::new();
        let interval: f32 = 1.0 / request_rate;
        for i in 0..num_requests {
            let delay: f32 = i as f32 * interval;
            v.push(delay);
        }
        v
    }

    match request_distribution {
        RequestDistribution::Poisson(request_rate) => {
            get_poisson_delays(*request_rate, num_requests)
        }
        RequestDistribution::Even(request_rate) => get_even_delays(*request_rate, num_requests),
        RequestDistribution::Same() => vec![0.0; num_requests],
    }
}

/// Generates vectors of prompt and decode lengths with uniform distributions
fn get_distribution(
    prompt_range: Range<u32>,
    decode_range: Range<u32>,
    num_requests: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut prompt: Vec<u32> = Vec::new();
    let mut decode: Vec<u32> = Vec::new();
    let prompt_dist: Uniform<u32> = Uniform::from(prompt_range);
    let decode_dist: Uniform<u32> = Uniform::from(decode_range);
    for _i in 0..num_requests {
        let p: u32 = prompt_dist.sample(&mut rand::thread_rng());
        let d: u32 = decode_dist.sample(&mut rand::thread_rng());
        prompt.push(p);
        decode.push(d);
    }
    (prompt, decode)
}

/// Sends a request to the cserve server and returns the time it took to get a response
async fn send_req(
    framework: &str,
    request_id: usize,
    delay: f32,
    prompt: String,
    prompt_len: u32,
    decode_len: u32,
    best_of: u32,
    use_beam_search: bool,
    port: u32,
    hostname: &str,
    verbosity: bool,
) -> ResponseStatistics {
    let mut headers: HeaderMap = HeaderMap::new();
    headers.insert("Content-Type", "application/json".parse().unwrap());

    let temperature: f32 = match use_beam_search {
        true => 0.0,
        false => 1.0,
    };

    let body = match framework {
        "cserve" => Framework::Cserve(CserveBody {
            prompt: prompt,
            sampling_params: SamplingParams {
                n: 1,
                max_tokens: decode_len,
                best_of: best_of,
                use_beam_search: use_beam_search,
                temperature: temperature,
                ignore_eos: true,
                top_p: 1.0,
            },
            stream: false,
        }),
        "vllm" => Framework::Vllm(VllmBody {
            prompt: prompt,
            stream: false,
            n: 1,
            max_tokens: decode_len,
            temperature: temperature,
            top_p: 1.0,
            ignore_eos: true,
            use_beam_search: use_beam_search,
            best_of: best_of,
        }),
        _ => panic!("Invalid framework"),
    };

    let client: reqwest::Client = reqwest::Client::new();

    sleep(Duration::from_secs_f32(delay)).await;
    let start: std::time::Instant = std::time::Instant::now();

    let resp: reqwest::Response = match body {
        Framework::Cserve(body) => client
            .post(format!("http://{hostname}:{port}/cserve/v1/generate"))
            .headers(headers)
            .json(&body)
            .send()
            .await
            .expect("Failed to get response"),
        Framework::Vllm(body) => client
            .post(format!("http://{hostname}:{port}/generate"))
            .headers(headers)
            .json(&body)
            .send()
            .await
            .expect("Failed to get response"),
    };
    let status = resp.status(); // Clone the response status
    let text: String = (resp).text().await.expect("Failed to get text");
    let duration: Duration = start.elapsed();

    if verbosity {
        if !status.is_success() {
            println!(
                "{}{:>5}: with CODE {}",
                "Request".cyan().bold(),
                request_id.to_string().purple().bold(),
                status.canonical_reason().unwrap().yellow().bold()
            );
        } else {
            let output_text: HashMap<String, Vec<String>> =
                serde_json::from_str(text.as_str()).unwrap();
            let completion = output_text["text"][0].replace("\n", " ").replace("\r", " ");
            let formatted_completion = match completion.chars().count() {
                0..=100 => completion,
                _ => format!(
                    "{}...{}",
                    &completion.chars().take(50).collect::<String>(),
                    &completion
                        .chars()
                        .skip(completion.chars().count() - 50)
                        .collect::<String>()
                ),
            };
            println!(
                "{}{:>5}: <<{}>> with CODE {} has Delay: {:.7}s, Latency: {:.7}s, Prompt Len: {}, Decode Len: {}",
                "Request".cyan().bold(),
                request_id.to_string().purple().bold(),
                formatted_completion,
                status.canonical_reason().unwrap().yellow().bold(),
                format!("{:.2}", delay).green().bold(),
                format!("{:.2}", duration.as_secs_f32()).green().bold(),
                prompt_len.to_string().green().bold(),
                decode_len.to_string().green().bold()
            );
        }
    }

    ResponseStatistics {
        latency: duration,
        latency_per_token: duration / (prompt_len + decode_len),
        latency_per_output_token: duration / decode_len,
        success: status.is_success(),
    }
}

async fn run_requests(
    framework: &str,
    num_requests: usize,
    request_distribution: &RequestDistribution,
    port: u32,
    hostname: &str,
    prompt_range: Range<u32>,
    decode_range: Range<u32>,
    best_of: u32,
    use_beam_search: bool,
    input_type: &InputType,
    tokenizer: &Tokenizer,
    verbose: bool,
) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
    let (prompt_lens, decode_lens): (Vec<u32>, Vec<u32>) =
        get_distribution(prompt_range, decode_range, num_requests);
    let prompts = prompt_lens
        .iter()
        .map(|&p| tokenize(&tokenizer, input_type, p as usize, verbose));

    let delays: Vec<f32> = get_delays(num_requests, &request_distribution);
    // Send requests
    let mut futures = Vec::new();
    for (i, prompt) in prompts.enumerate() {
        futures.push(send_req(
            framework,
            i,
            delays[i],
            prompt.0,
            prompt.1 as u32,
            decode_lens[i],
            best_of,
            use_beam_search,
            port,
            hostname,
            verbose,
        ));
    }
    let start: std::time::Instant = std::time::Instant::now();
    let results: Vec<ResponseStatistics> = join_all(futures).await;
    let duration: Duration = start.elapsed();
    let mut num_failed: usize = 0;
    for status in &results {
        if !status.success {
            num_failed += 1;
        }
    }
    if num_failed > 0 {
        let msg: String = format!("{} requests failed. DO NOT TRUST the results!", num_failed);
        return Err(msg.into());
    }

    // Compute statistics
    let mut total_latency: Duration = Duration::from_secs(0);
    let mut total_latency_per_token: Duration = Duration::from_secs(0);
    let mut total_latency_per_output_token: Duration = Duration::from_secs(0);
    for result in results {
        total_latency += result.latency;
        total_latency_per_token += result.latency_per_token;
        total_latency_per_output_token += result.latency_per_output_token;
    }
    let avg_latency: Duration = total_latency / num_requests as u32;
    let avg_latency_per_token: Duration = total_latency_per_token / num_requests as u32;
    let avg_latency_per_output_token: Duration =
        total_latency_per_output_token / num_requests as u32;

    let result: BenchmarkResults = BenchmarkResults {
        n: 1,
        total_time: duration.as_secs_f32(),
        throughput: num_requests as f32 / duration.as_secs_f32(),
        avg_latency: avg_latency.as_secs_f32(),
        avg_latency_per_token: avg_latency_per_token.as_secs_f32(),
        avg_latency_per_output_token: avg_latency_per_output_token.as_secs_f32(),
        note: "".to_string(),
    };
    Ok(result)
}

fn average_results(
    benchmark_results: Vec<BenchmarkResults>,
    benchmark_note: String,
) -> BenchmarkResults {
    let mut final_result = BenchmarkResults {
        n: 0,
        total_time: 0.0,
        throughput: 0.0,
        avg_latency: 0.0,
        avg_latency_per_token: 0.0,
        avg_latency_per_output_token: 0.0,
        note: benchmark_note,
    };
    for result in benchmark_results.into_iter() {
        final_result.n += result.n;
        final_result.total_time += result.total_time;
        final_result.throughput += result.throughput;
        final_result.avg_latency += result.avg_latency;
        final_result.avg_latency_per_token += result.avg_latency_per_token;
        final_result.avg_latency_per_output_token += result.avg_latency_per_output_token;
    }
    final_result.total_time /= final_result.n as f32;
    final_result.throughput /= final_result.n as f32;
    final_result.avg_latency /= final_result.n as f32;
    final_result.avg_latency_per_token /= final_result.n as f32;
    final_result.avg_latency_per_output_token /= final_result.n as f32;
    final_result
}

fn merge_configs(file_config: Args, cli_config: Args) -> Args {
    let mut merged_config = Args::default();
    merged_config.text_file = cli_config.text_file.or(file_config.text_file);
    merged_config.tokenizer_name = cli_config.tokenizer_name.or(file_config.tokenizer_name);
    merged_config.request_distribution = cli_config
        .request_distribution
        .or(file_config.request_distribution);
    merged_config.num_requests = cli_config.num_requests.or(file_config.num_requests);
    merged_config.request_rate = cli_config.request_rate.or(file_config.request_rate);
    merged_config.prompt_low = cli_config.prompt_low.or(file_config.prompt_low);
    merged_config.prompt_high = cli_config.prompt_high.or(file_config.prompt_high);
    merged_config.decode_low = cli_config.decode_low.or(file_config.decode_low);
    merged_config.decode_high = cli_config.decode_high.or(file_config.decode_high);
    merged_config.port = cli_config.port.or(file_config.port);
    merged_config.hostname = cli_config.hostname.or(file_config.hostname);
    merged_config.best_of = cli_config.best_of.or(file_config.best_of);
    merged_config.framework = cli_config.framework.or(file_config.framework);
    merged_config.use_beam_search = cli_config.use_beam_search.or(file_config.use_beam_search);
    merged_config.num_samples = cli_config.num_samples.or(file_config.num_samples);
    merged_config.benchmark_note = cli_config.benchmark_note.or(file_config.benchmark_note);
    merged_config.output_file = cli_config.output_file.or(file_config.output_file);
    merged_config.verbose = cli_config.verbose || file_config.verbose;
    merged_config
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input_args: Args = Args::parse();

    let args: Args = if let Some(config_file) = input_args.config_file.clone() {
        let file = File::open(config_file)?;
        let reader = BufReader::new(file);
        let file_config = serde_json::from_reader(reader)?;
        merge_configs(file_config, input_args)
    } else {
        input_args
    };

    // Do Validation and set arguments
    let num_samples: usize = if let Some(num_samples) = args.num_samples {
        if num_samples == 0 {
            return Err("Number of samples must be greater than 0".into());
        }
        println!(
            "{} Number of samples: {}",
            "INFO".blue().bold(),
            num_samples.to_string().green().bold()
        );
        num_samples
    } else {
        println!(
            "{} Number of samples not specified. Defaulting to 1",
            "WARNING".yellow().bold()
        );
        1
    };

    let num_requests: usize = if let Some(num_requests) = args.num_requests {
        if num_requests == 0 {
            return Err("Number of requests must be greater than 0".into());
        }
        println!(
            "{} Number of requests: {}",
            "INFO".blue().bold(),
            num_requests.to_string().green().bold()
        );
        num_requests
    } else {
        println!(
            "{} Number of requests not specified. Defaulting to 1",
            "WARNING".yellow().bold()
        );
        1
    };

    let best_of: u32 = if let Some(best_of) = args.best_of {
        if best_of == 0 {
            return Err("Best of must be greater than 0".into());
        }
        println!(
            "{} Best of: {}",
            "INFO".blue().bold(),
            best_of.to_string().green().bold()
        );
        best_of
    } else {
        println!(
            "{} Best of not specified. Defaulting to 1",
            "WARNING".yellow().bold()
        );
        1
    };

    let use_beam_search: bool = if let Some(use_beam_search) = args.use_beam_search {
        println!(
            "{} Use Beam Search: {}",
            "INFO".blue().bold(),
            use_beam_search.to_string().green().bold()
        );
        use_beam_search
    } else {
        println!(
            "{} Use Beam Search not specified. Defaulting to false",
            "WARNING".yellow().bold()
        );
        false
    };

    let request_rate: f32 = if let Some(request_rate) = args.request_rate {
        if request_rate < 0.0 {
            return Err("Request rate must be greater than 0".into());
        }
        println!(
            "{} Request rate: {}",
            "INFO".blue().bold(),
            request_rate.to_string().green().bold()
        );
        request_rate
    } else {
        println!(
            "{} Request rate not specified. Defaulting to 1.0",
            "WARNING".yellow().bold()
        );
        1.0
    };

    let port: u32 = if let Some(port) = args.port {
        println!(
            "{} Port: {}",
            "INFO".blue().bold(),
            port.to_string().green().bold()
        );
        port
    } else {
        println!(
            "{} Port not specified. Defaulting to 8000",
            "WARNING".yellow().bold()
        );
        8000
    };

    let hostname: String = if let Some(hostname) = args.hostname {
        println!(
            "{} Hostname: {}",
            "INFO".blue().bold(),
            hostname.green().bold()
        );
        hostname
    } else {
        println!(
            "{} Hostname not specified. Defaulting to localhost",
            "WARNING".yellow().bold()
        );
        "localhost".to_string()
    };

    let request_distribution: RequestDistribution =
        if let Some(distribution) = args.request_distribution {
            match distribution.as_str() {
                "poisson" => RequestDistribution::Poisson(request_rate),
                "even" => RequestDistribution::Even(request_rate),
                "same" => RequestDistribution::Same(),
                _ => panic!("Invalid request distribution: {}", distribution),
            }
        } else {
            println!(
                "{} Request distribution not specified. Defaulting to same",
                "WARNING".yellow().bold()
            );
            RequestDistribution::Same()
        };
    println!(
        "{} Request Distribution: {}",
        "INFO".blue().bold(),
        format!("{:?}", request_distribution).green().bold()
    );

    let tokenizer_name = if let Some(tokenizer_name) = args.tokenizer_name.clone() {
        println!(
            "{} Tokenizer Name: {}",
            "INFO".blue().bold(),
            tokenizer_name.green().bold()
        );
        tokenizer_name
    } else {
        return Err("Tokenizer name must be specified".into());
    };

    let framework = if let Some(framework) = args.framework {
        if framework != "cserve" && framework != "vllm" {
            return Err("Invalid framework".into());
        }
        println!(
            "{} Framework: {}",
            "INFO".blue().bold(),
            framework.green().bold()
        );
        framework
    } else {
        return Err("Framework must be specified".into());
    };

    let benchmark_note: String = if let Some(benchmark_note) = args.benchmark_note {
        benchmark_note
    } else {
        "".to_string()
    };

    // Set up request distributions
    let prompt_low: u32 = if let Some(prompt_low) = args.prompt_low {
        println!(
            "{} Prompt Low: {}",
            "INFO".blue().bold(),
            prompt_low.to_string().green().bold()
        );
        prompt_low
    } else {
        println!(
            "{} Prompt Low not specified. Defaulting to 1",
            "WARNING".yellow().bold()
        );
        1
    };
    let prompt_high: u32 = if let Some(prompt_high) = args.prompt_high {
        println!(
            "{} Prompt High: {}",
            "INFO".blue().bold(),
            prompt_high.to_string().green().bold()
        );
        prompt_high
    } else {
        println!(
            "{} Prompt High not specified. Defaulting to 1024",
            "WARNING".yellow().bold()
        );
        1024
    };
    let decode_low: u32 = if let Some(decode_low) = args.decode_low {
        println!(
            "{} Decode Low: {}",
            "INFO".blue().bold(),
            decode_low.to_string().green().bold()
        );
        decode_low
    } else {
        println!(
            "{} Decode Low not specified. Defaulting to 1",
            "WARNING".yellow().bold()
        );
        1
    };
    let decode_high: u32 = if let Some(decode_high) = args.decode_high {
        println!(
            "{} Decode High: {}",
            "INFO".blue().bold(),
            decode_high.to_string().green().bold()
        );
        decode_high
    } else {
        println!(
            "{} Decode High not specified. Defaulting to 1024",
            "WARNING".yellow().bold()
        );
        1024
    };
    if prompt_high <= prompt_low {
        return Err("Prompt high must be greater than prompt low".into());
    }
    if decode_high <= decode_low {
        return Err("Decode high must be greater than decode low".into());
    }
    let prompt_range: Range<u32> = prompt_low..prompt_high;
    let decode_range: Range<u32> = decode_low..decode_high;
    println!(
        "{} Prompt Range: {} to {}",
        "INFO".blue().bold(),
        prompt_low.to_string().green().bold(),
        prompt_high.to_string().green().bold()
    );
    println!(
        "{} Decode Range: {} to {}",
        "INFO".blue().bold(),
        decode_low.to_string().green().bold(),
        decode_high.to_string().green().bold()
    );

    let tokenizer: Tokenizer =
        Tokenizer::from_pretrained(tokenizer_name, None).expect("Failed to load Tokenizer");

    // Read text file or random input set
    let input_type: InputType = if let Some(filename) = args.text_file {
        if filename.is_empty() {
            println!(
                "{} No text file specified. Defaulting to random input",
                "WARNING".yellow().bold()
            );
            InputType::Random(tokenizer.get_vocab_size(false) as u32)
        } else {
            println!(
                "{} Text file: {}",
                "INFO".blue().bold(),
                filename.green().bold()
            );
            let mut file: File = File::open(filename)?;
            let mut contents: String = String::new();
            file.read_to_string(&mut contents)?;
            InputType::File(contents)
        }
    } else {
        println!(
            "{} No text file specified. Defaulting to random input",
            "WARNING".yellow().bold()
        );
        InputType::Random(tokenizer.get_vocab_size(false) as u32)
    };
    println!("-------------------------------------------------");

    let mut results: Vec<BenchmarkResults> = Vec::new();
    for _ in 0..num_samples {
        let result: BenchmarkResults = run_requests(
            &framework,
            num_requests,
            &request_distribution,
            port,
            &hostname,
            prompt_range.clone(),
            decode_range.clone(),
            best_of,
            use_beam_search,
            &input_type,
            &tokenizer,
            args.verbose,
        )
        .await?;
        results.push(result);
    }

    let final_result: BenchmarkResults = average_results(results, benchmark_note);

    if let Some(output_file) = args.output_file {
        let file = File::create(output_file)?;
        serde_json::to_writer_pretty(file, &final_result)?;
    } else {
        println!("{} {}", "INFO".blue().bold(), "Benchmark Results");
        println!("{}", serde_json::to_string_pretty(&final_result)?);
    }

    Ok(())
}
