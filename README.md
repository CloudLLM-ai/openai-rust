# openai-rust2

[![Crates.io](https://img.shields.io/crates/v/openai-rust2)](https://crates.io/crates/openai-rust2)
[![docs.rs](https://img.shields.io/docsrs/openai-rust2)](https://docs.rs/openai-rust2)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Unofficial async Rust client for the [OpenAI API](https://platform.openai.com/docs/api-reference) and **OpenAI-compatible** endpoints (xAI Grok, Ollama, OpenRouter, self-hosted gateways, …).

This is a maintained fork published as **`openai-rust2`** (original: [openai-rust](https://crates.io/crates/openai-rust)).

## Features

- [x] [List models](https://platform.openai.com/docs/api-reference/models/list)
- [x] [Chat Completions](https://platform.openai.com/docs/api-reference/chat/create) (+ streaming)
- [x] [Completions](https://platform.openai.com/docs/api-reference/completions/create) (legacy)
- [x] [Embeddings](https://platform.openai.com/docs/api-reference/embeddings/create)
- [x] [Images](https://platform.openai.com/docs/api-reference/images)
- [x] xAI Agent Tools / Responses API (`create_responses`)
- [x] OpenAI Responses API tools (`web_search`, `file_search`, `code_interpreter`)
- [x] Shared connection-pooled HTTP client + application-level retry on transient failures
- [ ] Audio / Files / Moderations / Fine-tuning

## Quick start

```rust,ignore
use openai_rust2 as openai_rust;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = openai_rust::Client::new(&std::env::var("OPENAI_API_KEY")?);
    let args = openai_rust::chat::ChatArguments::new(
        "gpt-4o-mini",
        vec![openai_rust::chat::Message {
            role: "user".into(),
            content: "Hello!".into(),
        }],
    );
    let res = client.create_chat(args, None).await?;
    println!("{}", res);
    Ok(())
}
```

Local Ollama (or any OpenAI-compatible base URL):

```rust,ignore
use openai_rust2 as openai_rust;

let client = openai_rust::Client::new_with_base_url(
    "", // many local servers ignore the key
    "http://localhost:11434",
);
```

Run the chat example:

```bash
OPENAI_API_KEY=sk-... cargo run --example chat
```

## Development

```bash
make clippy   # fmt + clippy -D warnings
make test     # unit tests always; live tests skip without OPENAI_API_KEY
```

## Projects using this crate

- [cloudllm](https://github.com/CloudLLM-ai/cloudllm) — agent toolkit (LLMSession, multi-provider clients, MentisDB)
- [uninews](https://github.com/CloudLLM-ai/uninews) — universal news scraper (via cloudllm)

## License

MIT
