use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug, Clone)]
pub struct ImageArguments {
    /// A text description of the desired image(s). The maximum length is 1000 characters.
    pub prompt: String,
    /// The model to use for image generation (e.g., "gpt-image-1").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The number of images to generate. Must be between 1 and 10. Defaults to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// The size of the generated images. Must be one of "1024x1024", "1024x1536", or "1536x1024". Defaults to "1024x1024".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// The quality of the generated images. Must be "standard" or "hd". Defaults to "standard".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl ImageArguments {
    pub fn new(prompt: impl AsRef<str>) -> Self {
        Self {
            prompt: prompt.as_ref().to_owned(),
            model: None,
            n: None,
            size: None,
            quality: None,
            user: None,
        }
    }
}

#[derive(Deserialize, Debug)]
pub(crate) enum ImageObject {
    #[serde(alias = "url")]
    Url(String),
    #[serde(alias = "b64_json")]
    Base64JSON(String),
}

#[derive(Deserialize, Debug)]
pub(crate) struct ImageResponse {
    #[allow(dead_code)]
    created: u32,
    pub data: Vec<ImageObject>,
}
