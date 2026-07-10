//! Integration tests against live OpenAI-compatible APIs.
//!
//! All tests that hit the network skip cleanly when `OPENAI_API_KEY` is unset.
//! The optional Ollama probe only runs when `http://localhost:11434` is reachable.

use base64::{engine::general_purpose, Engine};
use futures_util::StreamExt;
use openai_rust2 as openai_rust;
use std::fs::File;
use std::io::Write;
use std::time::Duration;

fn required_env_or_skip(key: &str) -> Option<String> {
    match std::env::var(key) {
        Ok(value) if !value.is_empty() => Some(value),
        _ => {
            eprintln!("Skipping live test because {key} is not set");
            None
        }
    }
}

async fn ollama_reachable() -> bool {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .ok();
    let Some(client) = client else {
        return false;
    };
    client
        .get("http://localhost:11434/api/tags")
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

#[tokio::test]
pub async fn list_models() {
    let Some(key) = required_env_or_skip("OPENAI_API_KEY") else {
        return;
    };

    let c = openai_rust::Client::new(&key);
    let models_vec = c.list_models(None).await.expect("list_models default base");
    assert!(!models_vec.is_empty());

    let c_openai_manually = openai_rust::Client::new_with_base_url(&key, "https://api.openai.com");
    let models_vec = c_openai_manually
        .list_models(None)
        .await
        .expect("list_models explicit openai base");
    assert!(!models_vec.is_empty());

    if ollama_reachable().await {
        let c_local_ollama = openai_rust::Client::new_with_base_url("", "http://localhost:11434");
        match c_local_ollama.list_models(None).await {
            Ok(models) => {
                assert!(!models.is_empty());
                for m in &models {
                    println!("Local Ollama Model: {}", m.id);
                }
            }
            Err(e) => {
                eprintln!("Skipping Ollama list_models probe: {e}");
            }
        }
    } else {
        eprintln!("Skipping Ollama list_models probe (localhost:11434 not reachable)");
    }
}

#[tokio::test]
pub async fn create_chat() {
    let Some(key) = required_env_or_skip("OPENAI_API_KEY") else {
        return;
    };
    let c = openai_rust::Client::new(&key);
    let args = openai_rust::chat::ChatArguments::new(
        "gpt-4o-mini",
        vec![openai_rust::chat::Message {
            role: "user".to_owned(),
            content: "Reply with exactly: pong".to_owned(),
        }],
    );
    let res = c.create_chat(args, None).await.expect("create_chat");
    assert!(!res.choices.is_empty());
}

#[tokio::test]
pub async fn create_chat_stream() {
    let Some(key) = required_env_or_skip("OPENAI_API_KEY") else {
        return;
    };
    let c = openai_rust::Client::new(&key);
    let args = openai_rust::chat::ChatArguments::new(
        "gpt-4o-mini",
        vec![openai_rust::chat::Message {
            role: "user".to_owned(),
            content: "Say hi in three words.".to_owned(),
        }],
    );

    let chunks = c
        .create_chat_stream(args, None)
        .await
        .expect("create_chat_stream")
        .collect::<Vec<_>>()
        .await;
    assert!(!chunks.is_empty());
}

#[tokio::test]
pub async fn create_completion() {
    let Some(key) = required_env_or_skip("OPENAI_API_KEY") else {
        return;
    };
    let c = openai_rust::Client::new(&key);
    let args = openai_rust::completions::CompletionArguments::new(
        "gpt-3.5-turbo-instruct",
        "The quick brown fox".to_owned(),
    );
    c.create_completion(args, None)
        .await
        .expect("create_completion");
}

#[tokio::test]
pub async fn create_completion_logprobs() {
    let Some(key) = required_env_or_skip("OPENAI_API_KEY") else {
        return;
    };
    let c = openai_rust::Client::new(&key);
    let mut args = openai_rust::completions::CompletionArguments::new(
        "gpt-3.5-turbo-instruct",
        "The quick brown fox".to_owned(),
    );
    args.logprobs = Some(1);
    c.create_completion(args, None)
        .await
        .expect("create_completion_logprobs");
}

#[tokio::test]
pub async fn create_embeddings() {
    let Some(key) = required_env_or_skip("OPENAI_API_KEY") else {
        return;
    };
    let c = openai_rust::Client::new(&key);
    let args = openai_rust::embeddings::EmbeddingsArguments::new(
        "text-embedding-3-small",
        "The food was delicious and the waiter...".to_owned(),
    );
    c.create_embeddings(args, None)
        .await
        .expect("create_embeddings");
}

#[tokio::test]
pub async fn external_client() {
    let Some(key) = required_env_or_skip("OPENAI_API_KEY") else {
        return;
    };
    let req_c = reqwest::ClientBuilder::new()
        .user_agent("openai-rust2-tests")
        .build()
        .unwrap();
    let c = openai_rust::Client::new_with_client(&key, req_c);
    c.list_models(None)
        .await
        .expect("external_client list_models");
}

#[tokio::test]
pub async fn create_image() {
    let Some(key) = required_env_or_skip("OPENAI_API_KEY") else {
        return;
    };
    let c = openai_rust::Client::new(&key);
    let args = openai_rust::images::ImageArguments::new(
        "A simple geometric red circle on a white background, flat design.",
    );
    let base64_images = c.create_image(args, None).await.expect("create_image");

    if let Some(base64_image) = base64_images.first() {
        let image_bytes = general_purpose::STANDARD
            .decode(base64_image)
            .expect("decode image base64");
        let mut file = File::create("generated_image.png").expect("create png file");
        file.write_all(&image_bytes).expect("write png");
    }
}

#[test]
pub fn create_chat_retries_are_configurable() {
    // Pure construction check — no network.
    let _client = openai_rust::Client::new("test-key").with_max_retries(0);
    let _shared = openai_rust::Client::shared_client();
}
