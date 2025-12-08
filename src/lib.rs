pub extern crate futures_util;
use anyhow::{anyhow, Result};
use lazy_static::lazy_static;

lazy_static! {
    static ref DEFAULT_BASE_URL: reqwest::Url =
        reqwest::Url::parse("https://api.openai.com/v1/models").unwrap();
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
    pub fn new(api_key: &str) -> Client {
        let req_client = reqwest::ClientBuilder::new().build().unwrap();
        Client {
            req_client,
            key: api_key.to_owned(),
            base_url: DEFAULT_BASE_URL.clone(),
        }
    }

    pub fn new_with_client(api_key: &str, req_client: reqwest::Client) -> Client {
        Client {
            req_client,
            key: api_key.to_owned(),
            base_url: DEFAULT_BASE_URL.clone(),
        }
    }

    pub fn new_with_base_url(api_key: &str, base_url: &str) -> Client {
        let req_client = reqwest::ClientBuilder::new().build().unwrap();
        let base_url = reqwest::Url::parse(base_url).unwrap();
        Client {
            req_client,
            key: api_key.to_owned(),
            base_url,
        }
    }

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

    pub async fn list_models(
        &self,
        opt_url_path: Option<String>,
    ) -> Result<Vec<models::Model>, anyhow::Error> {
        let mut url = self.base_url.clone();
        url.set_path(&opt_url_path.unwrap_or_else(|| String::from("/v1/models")));

        let res = self
            .req_client
            .get(url)
            .bearer_auth(&self.key)
            .send()
            .await?;

        if res.status() == 200 {
            Ok(res.json::<models::ListModelsResponse>().await?.data)
        } else {
            Err(anyhow!(res.text().await?))
        }
    }

    pub async fn create_chat(
        &self,
        args: chat::ChatArguments,
        opt_url_path: Option<String>,
    ) -> Result<chat::ChatCompletion, anyhow::Error> {
        let mut url = self.base_url.clone();
        url.set_path(&opt_url_path.unwrap_or_else(|| String::from("/v1/chat/completions")));

        let res = self
            .req_client
            .post(url)
            .bearer_auth(&self.key)
            .json(&args)
            .send()
            .await?;

        if res.status() == 200 {
            Ok(res.json().await?)
        } else {
            Err(anyhow!(res.text().await?))
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
            Err(anyhow!(res.text().await?))
        }
    }

    pub async fn create_completion(
        &self,
        args: completions::CompletionArguments,
        opt_url_path: Option<String>,
    ) -> Result<completions::CompletionResponse> {
        let mut url = self.base_url.clone();
        url.set_path(&opt_url_path.unwrap_or_else(|| String::from("/v1/completions")));

        let res = self
            .req_client
            .post(url)
            .bearer_auth(&self.key)
            .json(&args)
            .send()
            .await?;

        if res.status() == 200 {
            Ok(res.json().await?)
        } else {
            Err(anyhow!(res.text().await?))
        }
    }

    pub async fn create_embeddings(
        &self,
        args: embeddings::EmbeddingsArguments,
        opt_url_path: Option<String>,
    ) -> Result<embeddings::EmbeddingsResponse> {
        let mut url = self.base_url.clone();
        url.set_path(&opt_url_path.unwrap_or_else(|| String::from("/v1/embeddings")));

        let res = self
            .req_client
            .post(url)
            .bearer_auth(&self.key)
            .json(&args)
            .send()
            .await?;

        if res.status() == 200 {
            Ok(res.json().await?)
        } else {
            Err(anyhow!(res.text().await?))
        }
    }

    pub async fn create_image_old(
        &self,
        args: images::ImageArguments,
        opt_url_path: Option<String>,
    ) -> Result<Vec<String>> {
        let mut url = self.base_url.clone();
        url.set_path(&opt_url_path.unwrap_or_else(|| String::from("/v1/images/generations")));

        let res = self
            .req_client
            .post(url)
            .bearer_auth(&self.key)
            .json(&args)
            .send()
            .await?;

        if res.status() == 200 {
            Ok(res
                .json::<images::ImageResponse>()
                .await?
                .data
                .iter()
                .map(|o| match o {
                    images::ImageObject::Url(s) => s.to_string(),
                    images::ImageObject::Base64JSON(s) => s.to_string(),
                })
                .collect())
        } else {
            Err(anyhow!(res.text().await?))
        }
    }

    pub async fn create_image(
        &self,
        args: images::ImageArguments,
        opt_url_path: Option<String>,
    ) -> Result<Vec<String>> {
        let mut url = self.base_url.clone();
        url.set_path(&opt_url_path.unwrap_or_else(|| String::from("/v1/images/generations")));

        let image_args = images::ImageArguments {
            prompt: args.prompt,
            model: Some("gpt-image-1".to_string()),
            n: Some(1),
            size: Some("1024x1024".to_string()),
            quality: Some("auto".to_string()), // valid quality values are 'low', 'medium', 'high' and 'auto'
            //TODO: Make this an enum parameter to create_image
            user: None,
        };

        let res = self
            .req_client
            .post(url)
            .bearer_auth(&self.key)
            .json(&image_args)
            .send()
            .await?;

        if res.status() == 200 {
            Ok(res
                .json::<images::ImageResponse>()
                .await?
                .data
                .iter()
                .map(|o| match o {
                    images::ImageObject::Url(s) => s.to_string(),
                    images::ImageObject::Base64JSON(s) => s.to_string(),
                })
                .collect())
        } else {
            Err(anyhow!(res.text().await?))
        }
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
        let mut url = self.base_url.clone();
        url.set_path(&opt_url_path.unwrap_or_else(|| String::from("/v1/responses")));

        let res = self
            .req_client
            .post(url)
            .bearer_auth(&self.key)
            .json(&args)
            .send()
            .await?;

        if res.status() == 200 {
            Ok(res.json().await?)
        } else {
            Err(anyhow!(res.text().await?))
        }
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
        let mut url = self.base_url.clone();
        url.set_path(&opt_url_path.unwrap_or_else(|| String::from("/v1/responses")));

        let res = self
            .req_client
            .post(url)
            .bearer_auth(&self.key)
            .json(&args)
            .send()
            .await?;

        if res.status() == 200 {
            Ok(res.json().await?)
        } else {
            Err(anyhow!(res.text().await?))
        }
    }
}
