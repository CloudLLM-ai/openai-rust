// Here we will use the chat completion endpoint
use openai_rust::futures_util::StreamExt;
use openai_rust2 as openai_rust;
use std::io::Write;

#[tokio::main]
async fn main() {
    let client = openai_rust::Client::new(&std::env::var("OPENAI_API_KEY").unwrap());
    let args = openai_rust::chat::ChatArguments::new(
        "gpt-3.5-turbo",
        vec![openai_rust::chat::Message {
            role: "user".to_owned(),
            content: "Hello GPT!".to_owned(),
        }],
    );
    let mut res = client.create_chat_stream(args, None).await.unwrap();
    while let Some(chunk) = res.next().await {
        print!("{}", chunk.unwrap());
        std::io::stdout().flush().unwrap();
    }
}
