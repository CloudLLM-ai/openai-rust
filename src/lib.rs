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
    /// Per-call attempt budget. The shared HTTP client already does connection
    /// pooling + retries on the transport layer (see SHARED_HTTP_CLIENT);
    /// this knob is for application-level retries on top of that, used by
    /// `create_chat` for transient 200-stream body failures and 429/5xx.
    max_retries: u32,
}

impl Default for Client {
    fn default() -> Self {
        Self {
            req_client: SHARED_HTTP_CLIENT.clone(),
            key: String::new(),
            base_url: DEFAULT_BASE_URL.clone(),
            max_retries: 3,
        }
    }
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
        Self {
            req_client: SHARED_HTTP_CLIENT.clone(),
            key: api_key.to_owned(),
            base_url: DEFAULT_BASE_URL.clone(),
            max_retries: 3,
        }
    }

    /// Create a new client with a custom reqwest::Client.
    /// Use this when you need custom TLS, proxy, or connection pool settings.
    pub fn new_with_client(api_key: &str, req_client: reqwest::Client) -> Client {
        Self {
            req_client,
            key: api_key.to_owned(),
            base_url: DEFAULT_BASE_URL.clone(),
            max_retries: 3,
        }
    }

    /// Create a new client with the shared optimized HTTP client and custom base URL.
    pub fn new_with_base_url(api_key: &str, base_url: &str) -> Client {
        let base_url = reqwest::Url::parse(base_url).unwrap();
        Self {
            req_client: SHARED_HTTP_CLIENT.clone(),
            key: api_key.to_owned(),
            base_url,
            max_retries: 3,
        }
    }

    /// Create a new client with a custom reqwest::Client and custom base URL.
    pub fn new_with_client_and_base_url(
        api_key: &str,
        req_client: reqwest::Client,
        base_url: &str,
    ) -> Client {
        Self {
            req_client,
            key: api_key.to_owned(),
            base_url: reqwest::Url::parse(base_url).unwrap(),
            max_retries: 3,
        }
    }

    /// Get a reference to the shared HTTP client for advanced usage.
    pub fn shared_client() -> &'static reqwest::Client {
        &SHARED_HTTP_CLIENT
    }

    /// Override the per-call retry budget (default: 3). 0 disables retries.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Read the response body as text and deserialize it as JSON, surfacing
    /// the HTTP status and a (truncated) raw body in the error message.
    ///
    /// This is the path that every `create_*` / `list_models` method uses
    /// for the success branch. Routing everything through a single helper
    /// guarantees that callers never see a bare reqwest "error decoding
    /// response body" with no context, which previously made it impossible
    /// to tell whether a 200 response was actually a non-UTF-8 binary blob,
    /// a truncated stream, or some other transport-level failure.
    pub async fn read_and_parse_json<T: serde::de::DeserializeOwned>(
        res: reqwest::Response,
        error_context: &str,
    ) -> Result<T, anyhow::Error> {
        let status = res.status();
        match res.text().await {
            Ok(text) => serde_json::from_str(&text).map_err(|e| {
                anyhow!(
                    "{} failed to parse JSON response (status {}): {}. Raw body ({} bytes): {}",
                    error_context,
                    status,
                    e,
                    text.len(),
                    truncate_for_error(&text, 4096)
                )
            }),
            Err(e) => Err(anyhow!(
                "{} failed to read response body (status {}): {}",
                error_context,
                status,
                e
            )),
        }
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

        if res.status().is_success() {
            Self::read_and_parse_json(res, error_context).await
        } else {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            Err(anyhow!(
                "{} API error (status {}): {}",
                error_context,
                status,
                truncate_for_error(&body, 4096)
            ))
        }
    }

    /// Helper for GET requests.
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

        if res.status().is_success() {
            Self::read_and_parse_json(res, error_context).await
        } else {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            Err(anyhow!(
                "{} API error (status {}): {}",
                error_context,
                status,
                truncate_for_error(&body, 4096)
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
        // Chat completions can hit transient transport failures (connection
        // reset, truncated SSE → 200 stream, upstream 429/5xx). Retry on
        // the ones that are safe to retry, with exponential backoff and an
        // upper bound. The body-decode retry path is the one that fixed
        // the "error decoding response body" failure mode that previously
        // surfaced as a single fatal error.
        let mut attempt: u32 = 0;
        loop {
            let result = self.send_json::<chat::ChatCompletion>(&path, &args, "create_chat").await;
            match result {
                Ok(parsed) => return Ok(parsed),
                Err(e) => {
                    let is_transient = is_transient_error(&e);
                    if !is_transient || attempt >= self.max_retries {
                        return Err(e);
                    }
                    let backoff = backoff_for_attempt(attempt);
                    tokio::time::sleep(Duration::from_secs(backoff)).await;
                    attempt += 1;
                }
            }
        }
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
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            Err(anyhow!(
                "create_chat_stream failed: status={} body={}",
                status,
                truncate_for_error(&body, 4096)
            ))
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

/// Exponential backoff for `attempt` (0-based): 1, 2, 4, 8 … seconds,
/// capped at 30s so a runaway retry loop can never sleep for a minute
/// per attempt. Used for the body-decode retry path inside `create_chat`.
fn backoff_for_attempt(attempt: u32) -> u64 {
    (2u64.saturating_pow(attempt)).min(30)
}

/// Truncate a string for inclusion in error messages, marking the cut point
/// when bytes were dropped so logs don't get spammed by 10MB error bodies.
fn truncate_for_error(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        text.to_string()
    } else {
        let mut cut = max_bytes;
        while !text.is_char_boundary(cut) && cut > 0 {
            cut -= 1;
        }
        format!(
            "{}…[truncated, total {} bytes]",
            &text[..cut],
            text.len()
        )
    }
}

/// Heuristic: is the given `anyhow::Error` something that is safe to retry
/// on a fresh HTTP request? Used by `create_chat` to decide between
/// retrying the whole request and bubbling the error up.
fn is_transient_error(err: &anyhow::Error) -> bool {
    let msg = format!("{:#}", err);
    // Body-decode failures (reqwest's "error decoding response body") and
    // HTTP 429/5xx responses are the two retryable classes. Body-decode
    // failures show up here because `send_json` now embeds the reqwest
    // message verbatim: "create_chat failed to read response body (status
    // 200): error decoding response body".
    if msg.contains("error decoding response body") {
        return true;
    }
    if msg.contains("(status 429")
        || msg.contains("(status 500")
        || msg.contains("(status 502")
        || msg.contains("(status 503")
        || msg.contains("(status 504")
    {
        return true;
    }
    // reqwest connection-level errors look like
    // "error sending request" / "error decoding response body" / body stream
    // errors. Fall through to false for parse errors, 4xx, etc.
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    #[test]
    fn truncate_for_error_keeps_short_strings() {
        let s = "hello";
        assert_eq!(truncate_for_error(s, 10), "hello");
    }

    #[test]
    fn truncate_for_error_marks_cut_point() {
        let s = "a".repeat(100);
        let out = truncate_for_error(&s, 10);
        assert!(out.starts_with(&"a".repeat(10)));
        assert!(out.contains("truncated"));
        assert!(out.contains("100 bytes"));
    }

    #[test]
    fn truncate_for_error_respects_char_boundaries() {
        let s = "ááááá";
        let out = truncate_for_error(&s, 3);
        // Re-serializing must not panic on a split codepoint.
        let _ = std::str::from_utf8(out.as_bytes()).unwrap();
    }

    #[test]
    fn backoff_grows_then_caps() {
        assert_eq!(backoff_for_attempt(0), 1);
        assert_eq!(backoff_for_attempt(1), 2);
        assert_eq!(backoff_for_attempt(2), 4);
        assert_eq!(backoff_for_attempt(3), 8);
        assert_eq!(backoff_for_attempt(10), 30);
    }

    #[test]
    fn transient_classifier_recognises_known_signals() {
        assert!(is_transient_error(&anyhow!(
            "create_chat failed to read response body (status 200): error decoding response body"
        )));
        assert!(is_transient_error(&anyhow!(
            "create_chat API error (status 429): rate limited"
        )));
        assert!(is_transient_error(&anyhow!(
            "create_chat API error (status 503): unavailable"
        )));
        assert!(!is_transient_error(&anyhow!(
            "create_chat failed to parse JSON response (status 200): expected `,` at line 1"
        )));
        assert!(!is_transient_error(&anyhow!(
            "create_chat API error (status 401): unauthorized"
        )));
        assert!(!is_transient_error(&anyhow!(
            "create_chat API error (status 404): not found"
        )));
    }
}
