pub extern crate futures_util;
use anyhow::{anyhow, Result};
use lazy_static::lazy_static;
use std::time::Duration;

lazy_static! {
    static ref DEFAULT_BASE_URL: reqwest::Url =
        reqwest::Url::parse("https://api.openai.com/v1/models").unwrap();

    /// Shared HTTP client with optimized connection pooling for high-throughput LLM workloads.
    ///
    /// Configuration rationale:
    /// - pool_max_idle_per_host: 100 (handle burst traffic without connection churn)
    /// - pool_idle_timeout: 90s (keep connections warm between requests)
    /// - tcp_keepalive: 60s (detect dead connections proactively)
    /// - connect_timeout: 10s (fail fast on connection issues)
    /// - timeout: 300s (generous for large model responses)
    /// - tcp_nodelay: true (reduce latency for small requests)
    static ref SHARED_HTTP_CLIENT: reqwest::Client = {
        reqwest::ClientBuilder::new()
            .pool_idle_timeout(Some(Duration::from_secs(90)))
            .pool_max_idle_per_host(100)
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(10))
            .tcp_nodelay(true)
            .build()
            .expect("Failed to build shared HTTP client")
    };
}

pub struct Client {
    req_client: reqwest::Client,
    key: String,
    base_url: reqwest::Url,
}

pub mod chat;
pub mod completions;
pub mod edits;
pub mod embeddings;
pub mod images;
pub mod models;

impl Client {
    /// Create a new client with the shared optimized HTTP client.
    /// Uses connection pooling with keep-alive for high-throughput workloads.
    pub fn new(api_key: &str) -> Client {
        Client {
            req_client: SHARED_HTTP_CLIENT.clone(),
            key: api_key.to_owned(),
            base_url: DEFAULT_BASE_URL.clone(),
        }
    }

    /// Create a new client with a custom reqwest::Client.
    /// Use this when you need custom TLS, proxy, or connection pool settings.
    pub fn new_with_client(api_key: &str, req_client: reqwest::Client) -> Client {
        Client {
            req_client,
            key: api_key.to_owned(),
            base_url: DEFAULT_BASE_URL.clone(),
        }
    }

    /// Create a new client with the shared optimized HTTP client and custom base URL.
    pub fn new_with_base_url(api_key: &str, base_url: &str) -> Client {
        let base_url = reqwest::Url::parse(base_url).unwrap();
        Client {
            req_client: SHARED_HTTP_CLIENT.clone(),
            key: api_key.to_owned(),
            base_url,
        }
    }

    /// Create a new client with a custom reqwest::Client and custom base URL.
    pub fn new_with_client_and_base_url(
        api_key: &str,
        req_client: reqwest::Client,
        base_url: &str,
    ) -> Client {
        Client {
            req_client,
            key: api_key.to_owned(),
            base_url: reqwest::Url::parse(base_url).unwrap(),
        }
    }

    /// Get a reference to the shared HTTP client for advanced usage.
    pub fn shared_client() -> &'static reqwest::Client {
        &SHARED_HTTP_CLIENT
    }

    /// Helper to send a POST request with JSON body and parse the response.
    /// Returns the raw text for non-200 responses, or parses JSON for 200 responses.
    async fn send_json<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: &impl serde::Serialize,
        error_context: &str,
    ) -> Result<T, anyhow::Error> {
        let mut url = self.base_url.clone();
        url.set_path(path);

        let res = self
            .req_client
            .post(url)
            .bearer_auth(&self.key)
            .json(body)
            .send()
            .await?;

        let status = res.status();
        let text = res.text().await?;

        if status == 200 {
            serde_json::from_str(&text)
                .map_err(|e| anyhow!("{} failed to parse: {}. Raw: {}", error_context, e, text))
        } else {
            Err(anyhow!(
                "{} API error ({}): {}",
                error_context,
                status,
                text
            ))
        }
    }

    /// Helper for GET requests
    async fn send_get<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        error_context: &str,
    ) -> Result<T, anyhow::Error> {
        let mut url = self.base_url.clone();
        url.set_path(path);

        let res = self
            .req_client
            .get(url)
            .bearer_auth(&self.key)
            .send()
            .await?;

        let status = res.status();
        let text = res.text().await?;

        if status == 200 {
            serde_json::from_str(&text)
                .map_err(|e| anyhow!("{} failed to parse: {}. Raw: {}", error_context, e, text))
        } else {
            Err(anyhow!(
                "{} API error ({}): {}",
                error_context,
                status,
                text
            ))
        }
    }

    pub async fn list_models(
        &self,
        opt_url_path: Option<String>,
    ) -> Result<Vec<models::Model>, anyhow::Error> {
        let path = opt_url_path.unwrap_or_else(|| String::from("/v1/models"));
        #[derive(serde::Deserialize)]
        struct ListModelsResponse {
            data: Vec<models::Model>,
        }
        let response: ListModelsResponse = self.send_get(&path, "list_models").await?;
        Ok(response.data)
    }

    pub async fn create_chat(
        &self,
        args: chat::ChatArguments,
        opt_url_path: Option<String>,
    ) -> Result<chat::ChatCompletion, anyhow::Error> {
        let path = opt_url_path.unwrap_or_else(|| String::from("/v1/chat/completions"));
        self.send_json(&path, &args, "create_chat").await
    }

    pub async fn create_chat_stream(
        &self,
        args: chat::ChatArguments,
        opt_url_path: Option<String>,
    ) -> Result<chat::stream::ChatCompletionChunkStream> {
        let mut url = self.base_url.clone();
        url.set_path(&opt_url_path.unwrap_or_else(|| String::from("/v1/chat/completions")));

        let mut args = args;
        args.stream = Some(true);

        let res = self
            .req_client
            .post(url)
            .bearer_auth(&self.key)
            .json(&args)
            .send()
            .await?;

        if res.status() == 200 {
            Ok(chat::stream::ChatCompletionChunkStream::new(Box::pin(
                res.bytes_stream(),
            )))
        } else {
            Err(anyhow!(res.text().await?))
        }
    }

    pub async fn create_completion(
        &self,
        args: completions::CompletionArguments,
        opt_url_path: Option<String>,
    ) -> Result<completions::CompletionResponse> {
        let path = opt_url_path.unwrap_or_else(|| String::from("/v1/completions"));
        self.send_json(&path, &args, "create_completion").await
    }

    pub async fn create_embeddings(
        &self,
        args: embeddings::EmbeddingsArguments,
        opt_url_path: Option<String>,
    ) -> Result<embeddings::EmbeddingsResponse> {
        let path = opt_url_path.unwrap_or_else(|| String::from("/v1/embeddings"));
        self.send_json(&path, &args, "create_embeddings").await
    }

    pub async fn create_image_old(
        &self,
        args: images::ImageArguments,
        opt_url_path: Option<String>,
    ) -> Result<Vec<String>> {
        let path = opt_url_path.unwrap_or_else(|| String::from("/v1/images/generations"));
        let response: images::ImageResponse =
            self.send_json(&path, &args, "create_image_old").await?;
        Ok(response
            .data
            .iter()
            .map(|o| match o {
                images::ImageObject::Url(s) => s.to_string(),
                images::ImageObject::Base64JSON(s) => s.to_string(),
            })
            .collect())
    }

    pub async fn create_image(
        &self,
        args: images::ImageArguments,
        opt_url_path: Option<String>,
    ) -> Result<Vec<String>> {
        let path = opt_url_path.unwrap_or_else(|| String::from("/v1/images/generations"));
        let image_args = images::ImageArguments {
            prompt: args.prompt,
            model: Some("gpt-image-1".to_string()),
            n: Some(1),
            size: Some("1024x1024".to_string()),
            quality: Some("auto".to_string()),
            user: None,
        };
        let response: images::ImageResponse =
            self.send_json(&path, &image_args, "create_image").await?;
        Ok(response
            .data
            .iter()
            .map(|o| match o {
                images::ImageObject::Url(s) => s.to_string(),
                images::ImageObject::Base64JSON(s) => s.to_string(),
            })
            .collect())
    }

    /// Create a response using xAI's Responses API with agentic tool calling.
    ///
    /// This method calls the `/v1/responses` endpoint which supports server-side
    /// tools like web_search, x_search, code_execution, and more.
    ///
    /// # Arguments
    /// * `args` - The ResponsesArguments containing model, input messages, and tools
    /// * `opt_url_path` - Optional URL path override (defaults to `/v1/responses`)
    ///
    /// # Example
    /// ```rust,no_run
    /// use openai_rust2::chat::{ResponsesArguments, ResponsesMessage, GrokTool};
    /// use openai_rust2::Client;
    ///
    /// async fn example() -> anyhow::Result<()> {
    ///     let client = Client::new_with_base_url("your-api-key", "https://api.x.ai/v1");
    ///     let args = ResponsesArguments::new(
    ///         "grok-4-1-fast-reasoning",
    ///         vec![ResponsesMessage {
    ///             role: "user".to_string(),
    ///             content: "What is the current Bitcoin price?".to_string(),
    ///         }],
    ///     ).with_tools(vec![GrokTool::web_search()]);
    ///
    ///     let response = client.create_responses(args, None).await?;
    ///     println!("{}", response.get_text_content());
    ///     Ok(())
    /// }
    /// ```
    pub async fn create_responses(
        &self,
        args: chat::ResponsesArguments,
        opt_url_path: Option<String>,
    ) -> Result<chat::ResponsesCompletion, anyhow::Error> {
        let path = opt_url_path.unwrap_or_else(|| String::from("/v1/responses"));
        self.send_json(&path, &args, "create_responses").await
    }

    /// Create a response using OpenAI's Responses API with agentic tool calling.
    ///
    /// This method calls the `/v1/responses` endpoint which supports server-side
    /// tools like web_search, file_search, and code_interpreter.
    ///
    /// Supported models: gpt-5, gpt-4o, and other models with tool support.
    ///
    /// # Arguments
    /// * `args` - The OpenAIResponsesArguments containing model, input messages, and tools
    /// * `opt_url_path` - Optional URL path override (defaults to `/v1/responses`)
    ///
    /// # Example
    /// ```rust,no_run
    /// use openai_rust2::chat::{OpenAIResponsesArguments, ResponsesMessage, OpenAITool};
    /// use openai_rust2::Client;
    ///
    /// async fn example() -> anyhow::Result<()> {
    ///     let client = Client::new("your-openai-api-key");
    ///     let args = OpenAIResponsesArguments::new(
    ///         "gpt-5",
    ///         vec![ResponsesMessage {
    ///             role: "user".to_string(),
    ///             content: "What are the latest developments in AI?".to_string(),
    ///         }],
    ///     ).with_tools(vec![OpenAITool::web_search()]);
    ///
    ///     let response = client.create_openai_responses(args, None).await?;
    ///     println!("{}", response.get_text_content());
    ///     Ok(())
    /// }
    /// ```
    pub async fn create_openai_responses(
        &self,
        args: chat::OpenAIResponsesArguments,
        opt_url_path: Option<String>,
    ) -> Result<chat::ResponsesCompletion, anyhow::Error> {
        let path = opt_url_path.unwrap_or_else(|| String::from("/v1/responses"));
        self.send_json(&path, &args, "create_openai_responses")
            .await
    }
}
