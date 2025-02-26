use futures_util::StreamExt;
use lazy_static::lazy_static;
use openai_rust2 as openai_rust;
use std::env::var;

lazy_static! {
    static ref KEY: String = var("OPENAI_API_KEY").unwrap();
}

#[tokio::test]
pub async fn list_models() {
    let c = openai_rust::Client::new(&KEY);
    let models_vec = c.list_models(None).await.unwrap();
    assert!(models_vec.len() > 0);

    let c_openai_manually = openai_rust::Client::new_with_base_url(&KEY, "https://api.openai.com");
    let models_vec = c_openai_manually.list_models(None).await.unwrap();
    assert!(models_vec.len() > 0);

    let c_local_ollama = openai_rust::Client::new_with_base_url("", "http://localhost:11434");
    let models_vec = c_local_ollama.list_models(None).await.unwrap();
    assert!(models_vec.len() > 0);
    models_vec.iter().for_each(|m| {
        println!("Local Ollama Model: {}", m.id);
    });
}

#[tokio::test]
pub async fn create_chat() {
    let c = openai_rust::Client::new(&KEY);
    let args = openai_rust::chat::ChatArguments::new(
        "gpt-3.5-turbo",
        vec![openai_rust::chat::Message {
            role: "user".to_owned(),
            content: "Hello GPT!".to_owned(),
        }],
    );
    c.create_chat(args, None).await.unwrap();
}

#[tokio::test]
pub async fn create_chat_stream() {
    let c = openai_rust::Client::new(&KEY);
    let args = openai_rust::chat::ChatArguments::new(
        "gpt-3.5-turbo",
        vec![openai_rust::chat::Message {
            role: "user".to_owned(),
            content: "Hello GPT!".to_owned(),
        }],
    );

    c.create_chat_stream(args, None)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
}

#[tokio::test]
pub async fn create_completion() {
    let c = openai_rust::Client::new(&KEY);
    let args = openai_rust::completions::CompletionArguments::new(
        "text-davinci-003",
        "The quick brown fox".to_owned(),
    );
    c.create_completion(args, None).await.unwrap();
}

#[tokio::test]
pub async fn create_completion_logprobs() {
    let c = openai_rust::Client::new(&KEY);
    let mut args = openai_rust::completions::CompletionArguments::new(
        "text-davinci-003",
        "The quick brown fox".to_owned(),
    );
    args.logprobs = Some(1);
    c.create_completion(args, None).await.unwrap();
}

#[tokio::test]
pub async fn create_embeddings() {
    let c = openai_rust::Client::new(&KEY);
    let args = openai_rust::embeddings::EmbeddingsArguments::new(
        "text-embedding-ada-002",
        "The food was delicious and the waiter...".to_owned(),
    );
    c.create_embeddings(args, None).await.unwrap();
}

#[tokio::test]
pub async fn external_client() {
    use reqwest;
    let req_c = reqwest::ClientBuilder::new()
        .user_agent("My cool program")
        .build()
        .unwrap();
    let c = openai_rust::Client::new_with_client(&KEY, req_c);
    c.list_models(None).await.unwrap();
}

#[tokio::test]
pub async fn create_image() {
    let c = openai_rust::Client::new(&KEY);
    let args = openai_rust::images::ImageArguments::new("Lovecraftian Dagon");
    c.create_image(args, None).await.unwrap();
}
