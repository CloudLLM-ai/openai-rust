use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ResponseFormat {
    JsonObject,
    Text,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ImageGeneration {
    pub quality: Option<String>,       // e.g., "standard", "hd"
    pub size: Option<String>,          // e.g., "1024x1024"
    pub output_format: Option<String>, // e.g., "base64", "url"
}

#[derive(Serialize, Debug, Clone)]
pub struct ChatArguments {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_generation: Option<ImageGeneration>,
    /// xAI Agent Tools API - server-side tools for agentic capabilities.
    /// Includes: web_search, x_search, code_execution, collections_search, mcp.
    /// See: https://docs.x.ai/docs/guides/tools/overview
    #[serde(skip_serializing_if = "Option::is_none", rename = "server_tools")]
    pub grok_tools: Option<Vec<GrokTool>>,
}

impl ChatArguments {
    pub fn new(model: impl AsRef<str>, messages: Vec<Message>) -> ChatArguments {
        ChatArguments {
            model: model.as_ref().to_owned(),
            messages,
            temperature: None,
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            max_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
            response_format: None,
            image_generation: None,
            grok_tools: None,
        }
    }

    /// Add xAI server-side tools for agentic capabilities.
    /// Recommended model: `grok-4-1-fast` for best tool-calling performance.
    pub fn with_grok_tools(mut self, tools: Vec<GrokTool>) -> Self {
        self.grok_tools = Some(tools);
        self
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct ChatCompletion {
    #[serde(default)]
    pub id: Option<String>,
    pub created: u32,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub object: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

impl std::fmt::Display for ChatCompletion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.choices[0].message.content)?;
        Ok(())
    }
}

pub mod stream {
    use bytes::Bytes;
    use futures_util::Stream;
    use serde::Deserialize;
    use std::pin::Pin;
    use std::str;
    use std::task::Poll;

    #[derive(Deserialize, Debug, Clone)]
    pub struct ChatCompletionChunk {
        pub id: String,
        pub created: u32,
        pub model: String,
        pub choices: Vec<Choice>,
        pub system_fingerprint: Option<String>,
    }

    impl std::fmt::Display for ChatCompletionChunk {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "{}",
                self.choices[0].delta.content.as_ref().unwrap_or(&"".into())
            )?;
            Ok(())
        }
    }

    #[derive(Deserialize, Debug, Clone)]
    pub struct Choice {
        pub delta: ChoiceDelta,
        pub index: u32,
        pub finish_reason: Option<String>,
    }

    #[derive(Deserialize, Debug, Clone)]
    pub struct ChoiceDelta {
        pub content: Option<String>,
    }

    pub struct ChatCompletionChunkStream {
        byte_stream: Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>>>>,
        buf: String,
    }

    impl ChatCompletionChunkStream {
        pub(crate) fn new(stream: Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>>>>) -> Self {
            Self {
                byte_stream: stream,
                buf: String::new(),
            }
        }

        fn deserialize_buf(
            self: Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> Option<anyhow::Result<ChatCompletionChunk>> {
            let bufclone = self.buf.clone();
            let mut chunks = bufclone.split("\n\n").peekable();
            let first = chunks.next();
            let second = chunks.peek();

            match first {
                Some(first) => match first.strip_prefix("data: ") {
                    Some(chunk) => {
                        if !chunk.ends_with("}") {
                            None
                        } else {
                            if let Some(second) = second {
                                if second.ends_with("}") {
                                    cx.waker().wake_by_ref();
                                }
                            }
                            self.get_mut().buf = chunks.collect::<Vec<_>>().join("\n\n");
                            Some(
                                serde_json::from_str::<ChatCompletionChunk>(chunk)
                                    .map_err(|e| anyhow::anyhow!(e)),
                            )
                        }
                    }
                    None => None,
                },
                None => None,
            }
        }
    }

    impl Stream for ChatCompletionChunkStream {
        type Item = anyhow::Result<ChatCompletionChunk>;

        fn poll_next(
            mut self: Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> Poll<Option<Self::Item>> {
            if let Some(chunk) = self.as_mut().deserialize_buf(cx) {
                return Poll::Ready(Some(chunk));
            }

            match self.byte_stream.as_mut().poll_next(cx) {
                Poll::Ready(bytes_option) => match bytes_option {
                    Some(bytes_result) => match bytes_result {
                        Ok(bytes) => {
                            let data = str::from_utf8(&bytes)?.to_owned();
                            self.buf = self.buf.clone() + &data;
                            match self.deserialize_buf(cx) {
                                Some(chunk) => Poll::Ready(Some(chunk)),
                                None => {
                                    cx.waker().wake_by_ref();
                                    Poll::Pending
                                }
                            }
                        }
                        Err(e) => Poll::Ready(Some(Err(e.into()))),
                    },
                    None => Poll::Ready(None),
                },
                Poll::Pending => Poll::Pending,
            }
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Choice {
    #[serde(default)]
    pub index: Option<u32>,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

pub enum Role {
    System,
    Assistant,
    User,
}

// =============================================================================
// xAI Agent Tools API
// See: https://docs.x.ai/docs/guides/tools/overview
// =============================================================================

/// Represents a server-side tool available in xAI's Agent Tools API.
///
/// xAI provides agentic server-side tool calling where the model autonomously
/// explores, searches, and executes code. The server handles the entire
/// reasoning and tool-execution loop.
///
/// # Supported Models
/// - `grok-4-1-fast` (recommended for agentic tool calling)
/// - `grok-4-1-fast-non-reasoning`
/// - `grok-4`, `grok-4-fast`, `grok-4-fast-non-reasoning`
///
/// # Example
/// ```rust,no_run
/// use openai_rust2::chat::GrokTool;
///
/// let tools = vec![
///     GrokTool::web_search(),
///     GrokTool::x_search(),
///     GrokTool::code_execution(),
///     GrokTool::collections_search(vec!["collection-id-1".into()]),
///     GrokTool::mcp("https://my-mcp-server.com".into()),
/// ];
/// ```
#[derive(Serialize, Debug, Clone)]
pub struct GrokTool {
    /// The type of tool: "web_search", "x_search", "code_execution", "collections_search", "mcp"
    #[serde(rename = "type")]
    pub tool_type: GrokToolType,
    /// Restrict web search to specific domains (max 5). Only applies to web_search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,
    /// Inclusive start date for search results (ISO8601: YYYY-MM-DD). Applies to web_search and x_search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_date: Option<String>,
    /// Inclusive end date for search results (ISO8601: YYYY-MM-DD). Applies to web_search and x_search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_date: Option<String>,
    /// Collection IDs to search. Required for collections_search tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collection_ids: Option<Vec<String>>,
    /// MCP server URL. Required for mcp tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_url: Option<String>,
}

/// The type of xAI server-side tool.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GrokToolType {
    /// Real-time web search and page browsing
    WebSearch,
    /// Search X (Twitter) posts, users, and threads
    XSearch,
    /// Execute Python code for calculations and data analysis
    CodeExecution,
    /// Search uploaded document collections (knowledge bases)
    CollectionsSearch,
    /// Connect to external MCP servers for custom tools
    Mcp,
}

impl GrokTool {
    /// Create a web_search tool with default settings.
    /// Allows the agent to search the web and browse pages.
    pub fn web_search() -> Self {
        Self {
            tool_type: GrokToolType::WebSearch,
            allowed_domains: None,
            from_date: None,
            to_date: None,
            collection_ids: None,
            server_url: None,
        }
    }

    /// Create an x_search tool with default settings.
    /// Allows the agent to search X posts, users, and threads.
    pub fn x_search() -> Self {
        Self {
            tool_type: GrokToolType::XSearch,
            allowed_domains: None,
            from_date: None,
            to_date: None,
            collection_ids: None,
            server_url: None,
        }
    }

    /// Create a code_execution tool.
    /// Allows the agent to execute Python code for calculations and data analysis.
    pub fn code_execution() -> Self {
        Self {
            tool_type: GrokToolType::CodeExecution,
            allowed_domains: None,
            from_date: None,
            to_date: None,
            collection_ids: None,
            server_url: None,
        }
    }

    /// Create a collections_search tool with the specified collection IDs.
    /// Allows the agent to search through uploaded knowledge bases.
    pub fn collections_search(collection_ids: Vec<String>) -> Self {
        Self {
            tool_type: GrokToolType::CollectionsSearch,
            allowed_domains: None,
            from_date: None,
            to_date: None,
            collection_ids: Some(collection_ids),
            server_url: None,
        }
    }

    /// Create an MCP tool connecting to an external MCP server.
    /// Allows the agent to access custom tools from the specified server.
    pub fn mcp(server_url: String) -> Self {
        Self {
            tool_type: GrokToolType::Mcp,
            allowed_domains: None,
            from_date: None,
            to_date: None,
            collection_ids: None,
            server_url: Some(server_url),
        }
    }

    /// Restrict web search to specific domains (max 5).
    /// Only applies to web_search tool.
    pub fn with_allowed_domains(mut self, domains: Vec<String>) -> Self {
        self.allowed_domains = Some(domains);
        self
    }

    /// Set the date range for search results (ISO8601: YYYY-MM-DD).
    /// Applies to web_search and x_search tools.
    pub fn with_date_range(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.from_date = Some(from.into());
        self.to_date = Some(to.into());
        self
    }
}

// =============================================================================
// xAI Responses API
// See: https://docs.x.ai/docs/guides/tools/search-tools
// The Responses API is a separate endpoint (/v1/responses) for agentic tool calling.
// =============================================================================

/// Request arguments for xAI's Responses API endpoint (/v1/responses).
///
/// This API provides agentic tool calling where the model autonomously
/// explores, searches, and executes code. Unlike the Chat Completions API,
/// the Responses API uses `input` instead of `messages` and `tools` instead
/// of `server_tools`.
///
/// # Example
/// ```rust,no_run
/// use openai_rust2::chat::{ResponsesArguments, ResponsesMessage, GrokTool};
///
/// let args = ResponsesArguments::new(
///     "grok-4-1-fast-reasoning",
///     vec![ResponsesMessage {
///         role: "user".to_string(),
///         content: "What is the current price of Bitcoin?".to_string(),
///     }],
/// ).with_tools(vec![GrokTool::web_search()]);
/// ```
#[derive(Serialize, Debug, Clone)]
pub struct ResponsesArguments {
    pub model: String,
    pub input: Vec<ResponsesMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GrokTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
}

impl ResponsesArguments {
    /// Create new ResponsesArguments for the xAI Responses API.
    pub fn new(model: impl AsRef<str>, input: Vec<ResponsesMessage>) -> Self {
        Self {
            model: model.as_ref().to_owned(),
            input,
            tools: None,
            temperature: None,
            max_output_tokens: None,
        }
    }

    /// Add tools for agentic capabilities.
    pub fn with_tools(mut self, tools: Vec<GrokTool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set the temperature for response generation.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the maximum output tokens.
    pub fn with_max_output_tokens(mut self, max_tokens: u32) -> Self {
        self.max_output_tokens = Some(max_tokens);
        self
    }
}

/// Message format for the Responses API input array.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ResponsesMessage {
    pub role: String,
    pub content: String,
}

/// Response from xAI's Responses API.
///
/// The Responses API returns a different format from Chat Completions,
/// including citations for sources used during agentic search.
#[derive(Deserialize, Debug, Clone)]
pub struct ResponsesCompletion {
    #[serde(default)]
    pub id: Option<String>,
    /// The output content items from the model
    pub output: Vec<ResponsesOutputItem>,
    /// Citations for sources used during search (URLs)
    #[serde(default)]
    pub citations: Vec<String>,
    /// Token usage statistics
    pub usage: ResponsesUsage,
}

impl ResponsesCompletion {
    /// Extract the text content from the response output.
    pub fn get_text_content(&self) -> String {
        self.output
            .iter()
            .filter_map(|item| {
                if item.item_type == "message" {
                    item.content.as_ref().map(|contents| {
                        contents
                            .iter()
                            .filter_map(|c| {
                                if c.content_type == "output_text" {
                                    c.text.clone()
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("")
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

impl std::fmt::Display for ResponsesCompletion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.get_text_content())
    }
}

/// An output item in the Responses API response.
#[derive(Deserialize, Debug, Clone)]
pub struct ResponsesOutputItem {
    #[serde(rename = "type")]
    pub item_type: String,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<Vec<ResponsesContent>>,
}

/// Content within a Responses API output item.
#[derive(Deserialize, Debug, Clone)]
pub struct ResponsesContent {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(default)]
    pub text: Option<String>,
}

/// Token usage for Responses API.
#[derive(Deserialize, Debug, Clone)]
pub struct ResponsesUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}
