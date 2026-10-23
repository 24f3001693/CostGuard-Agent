import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import AWS_REGION, MODELS


class BedrockClient:
    """
    Amazon Bedrock Converse API wrapper.

    In mock mode, generate() returns a demo response and marks it as mock.
    In Bedrock mode, generate_result() returns the response text plus usage
    metadata when the provider supplies it.
    """

    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        self.client = None
        self.last_usage = None
        self.last_error = None

        if not use_mock:
            try:
                self.client = boto3.client(
                    "bedrock-runtime",
                    region_name=AWS_REGION,
                )
            except (BotoCoreError, Exception) as exc:
                self.last_error = str(exc)
                # Do not silently switch to mock mode. That could make a
                # real inference failure look like a successful AI response.
                raise RuntimeError(
                    f"Could not initialize Amazon Bedrock client in "
                    f"region {AWS_REGION}: {exc}"
                ) from exc

    def generate_result(self, prompt: str, model_type: str = "cheap") -> dict:
        """
        Return a structured result:
        {
            'text': str,
            'mode': 'mock' or 'bedrock',
            'usage': {'input_tokens': int|None, 'output_tokens': int|None},
            'error': str|None
        }

        Bedrock errors are returned as explicit errors, not as normal response
        text. The caller can decide whether to display/log them.
        """
        model_info = MODELS.get(model_type)
        if model_info is None:
            raise ValueError(f"Unknown model tier: {model_type}")

        self.last_usage = None
        self.last_error = None

        if self.use_mock:
            return {
                "text": self._mock_response(prompt, model_type, model_info),
                "mode": "mock",
                "usage": {
                    "input_tokens": None,
                    "output_tokens": None,
                },
                "error": None,
            }

        try:
            response = self.client.converse(
                modelId=model_info["id"],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ],
                inferenceConfig={
                    "maxTokens": 512,
                    "temperature": 0.3,
                },
            )

            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])
            text_parts = [
                block["text"]
                for block in content_blocks
                if isinstance(block, dict) and "text" in block
            ]
            response_text = "\n".join(text_parts).strip()

            usage_metadata = response.get("usage", {})
            usage = {
                "input_tokens": usage_metadata.get("inputTokens"),
                "output_tokens": usage_metadata.get("outputTokens"),
            }
            self.last_usage = usage

            if not response_text:
                return {
                    "text": "",
                    "mode": "bedrock",
                    "usage": usage,
                    "error": "Bedrock returned no text content.",
                }

            return {
                "text": response_text,
                "mode": "bedrock",
                "usage": usage,
                "error": None,
            }

        except (ClientError, BotoCoreError) as exc:
            self.last_error = str(exc)
            return {
                "text": "",
                "mode": "bedrock",
                "usage": None,
                "error": str(exc),
            }
        except Exception as exc:
            self.last_error = str(exc)
            return {
                "text": "",
                "mode": "bedrock",
                "usage": None,
                "error": f"Unexpected Bedrock response error: {exc}",
            }

    def generate(self, prompt: str, model_type: str = "cheap") -> str:
        """
        Backwards-compatible text-only method.

        For new code, prefer generate_result() so errors and usage metadata
        remain separate from the generated answer.
        """
        result = self.generate_result(prompt, model_type)

        if result["error"]:
            return f"[Bedrock Error] {result['error']}"

        return result["text"]

    def _mock_response(self, prompt: str, model_type: str, model_info: dict) -> str:
        """Demo-only response used while real Bedrock access is unavailable."""
        return (
            f"[MOCK - {model_info['name']}]\n\n"
            f"Aapne poocha: \"{prompt}\"\n\n"
            f"Yeh demo response {model_type} tier ke liye hai.\n"
            "Actual Bedrock access enable hone par real model response aayega.\n\n"
            f"Model: {model_info['name']}\n"
            "Usage tokens: unavailable in mock mode"
        )
