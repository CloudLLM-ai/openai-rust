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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_parameters: Option<SearchParameters>, // Grok-specific search parameter (for now)
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
            search_parameters: None, // Grok-specific search parameter (for now)
        }
    }

    pub fn with_search_parameters(mut self, params: SearchParameters) -> Self {
        self.search_parameters = Some(params);
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
                                serde_json::from_str::<ChatCompletionChunk>(&chunk)
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
            match self.as_mut().deserialize_buf(cx) {
                Some(chunk) => return Poll::Ready(Some(chunk)),
                None => {}
            };

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

// Grok-specific search parameters
#[derive(Serialize, Debug, Clone)]
pub struct SearchParameters {
    pub mode: SearchMode, // "off", "on", "auto" (Live search is enabled but model decides when to use it)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_citations: Option<bool>,
    /// Inclusive yyyy-mm-dd
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_date: Option<String>,
    /// Inclusive upper‐bound yyyy-mm-dd
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_date: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")] // <-- "On" → "on", etc.
pub enum SearchMode {
    On,
    Off,
    Auto,
}

impl SearchParameters {
    pub fn new(mode: SearchMode) -> Self {
        Self {
            mode,
            return_citations: None,
            from_date: None,
            to_date: None,
        }
    }
    pub fn with_citations(mut self, yes: bool) -> Self {
        self.return_citations = Some(yes);
        self
    }
    pub fn with_date_range_str(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.from_date = Some(from.into());
        self.to_date = Some(to.into());
        self
    }
}
