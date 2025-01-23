# openai-rust2

[![Test Status](https://github.com/LevitatingBusinessMan/openai-rust/workflows/Build/badge.svg)](https://github.com/LevitatingBusinessMan/openai-rust/actions)
[![Crates.io](https://img.shields.io/crates/v/openai-rust)](https://crates.io/crates/openai-rust)
[![docs.rs](https://img.shields.io/docsrs/openai-rust)](https://docs.rs/openai-rust/latest/openai_rust/)


This is an unofficial library to interact with the [Openai-API](https://platform.openai.com/docs/api-reference). The goal of this crate is to support the entire api while matching the official documentation as closely as possible.

#### Current features:
- [x] [Listing models](https://platform.openai.com/docs/api-reference/models/list)
- [x] [Completions](https://platform.openai.com/docs/api-reference/completions/create)
- [x] [Chat](https://platform.openai.com/docs/api-reference/chat/create)
- [x] [Streaming Chat](https://platform.openai.com/docs/api-reference/chat/create#chat/create-stream)
- [x] [Edit](https://platform.openai.com/docs/api-reference/edits/create)
- [x] [Embeddings](https://platform.openai.com/docs/api-reference/embeddings/create)
- [x] [Images](https://platform.openai.com/docs/api-reference/images)
- [ ] Audio
- [ ] Files
- [ ] Moderations
- [ ] Fine-tuning

### Example usage
```rust ignore
// Here we will use the chat completion endpoint connecting to openAI's default base URL
use openai_rust2 as openai_rust; // since this is a fork of openai_rust
let client = openai_rust::Client::new(&std::env::var("OPENAI_API_KEY").unwrap());
let args = openai_rust::chat::ChatArguments::new("gpt-3.5-turbo", vec![
    openai_rust::chat::Message {
        role: "user".to_owned(),
        content: "Hello GPT!".to_owned(),
    }
]);
let res = client.create_chat(args).await.unwrap();
println!("{}", res);
```

Here another example connecting to a local LLM server (Ollama's base URL)
```rust ignore
use openai_rust2 as openai_rust; // since this is a fork of openai_rust
let client = openai_rust::Client::new_with_base_url(
    "", // no need for an API key when connecting to a default ollama instance locally
    "http://localhost:11434"
);
```

You can run this code as an example with `OPENAI_API_KEY=(your key) cargo run --example chat`.

Checkout the examples directory for more usage examples. You can find documentation on [docs.rs](https://docs.rs/openai-rust/latest/openai_rust/).

### Projects using openai-rust
* [openai-cli](https://github.com/LevitatingBusinessMan/openai-cli): a CLI for interacting with GPT.
* [gpt-cli-rust](https://github.com/memochou1993/gpt-cli-rust): Another CLI.
* [electocracy](https://github.com/marioloko/electocracy): A digital voting system.
* [awsgpt](https://github.com/fizlip/awsgpt): Interact with the aws-cli via GPT.
