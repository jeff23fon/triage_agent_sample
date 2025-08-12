from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def make_settings_config(env_prefix: str = "") -> SettingsConfigDict:
    return SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix=env_prefix,
        extra="ignore",
        env_ignore_empty=True,
    )


class AzureOpenAISettings(BaseSettings):
    model_config = make_settings_config("AZURE_OPENAI_")
    endpoint: str = Field(..., alias="ENDPOINT")
    key: str = Field(..., alias="KEY")
    deployment: str = Field(..., alias="DEPLOYMENT")
    version: str = Field("", alias="VERSION")  # Optional

class AppSettings(BaseSettings):
    model_config = make_settings_config()
    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)  # type: ignore
