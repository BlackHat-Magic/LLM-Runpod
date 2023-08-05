# LLM-Runpod

Acts as a wrapper for LLMs compatible with HuggingFace's `transformers` library to provide a serverless, OpenAI-esque API via Runpod. May or may not end up in use for [Wayfarer](https://github.com/BlackHat-Magic/Wayfarer).

The code may not be directly usable depending on the LLM used, as different community-made models have different formats for the way that they recognize different messages in the input/output data. Check the documentation for whatever LLM you use and alter the strings used to generate the prompt accordingly.

Additionally, the code as-is assumes the very first message is a message that instructs the model on how to respond to the user. The idea was that this would be akin to a system message used for OpenAI's models. As a result, in the event that the conversation does not fit in the configured context window, it will remove the *second* message in the conversation until the conversation fits. If, for whatever reason, you do not want this behavior, and want to discard the first message so that the model only remembers the most recent messages, change thr `1` in `messages.pop(1)` to 0.