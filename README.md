# LLM-Runpod

Acts as a wrapper for LLMs compatible with HuggingFace's `transformers` library to provide a serverless, OpenAI-esque API via Runpod. May or may not end up in use for [Wayfarer](https://github.com/BlackHat-Magic/Wayfarer).

Unlike [Wayfarer-SD-Runpod](https://github.com/BlackHat-Magic/Wayfarer-SD-Runpod) (as of writing this; the SD repo needs an update), this requires that you set up a network volume and store the model there. The `TRANSFORMERS_CACHE` directory is the same as `CACHE_DIR` in the serverless worker. For some reason `transformers` likes ignoring environment variables when I use it so the cache dir is set manually in code.

The code may not be directly usable depending on the LLM used, as different community-made models have different formats for the way that they recognize different messages in the input/output data. Check the documentation for whatever LLM you use and alter the strings used to generate the prompt accordingly.

The `tokenizer.apply_chat_template()` method is intentionally not used because many models do not support it. If the one you intend to use does, go right ahead.

Additionally, the code as-is assumes the very first message is a message that instructs the model on how to respond to the user. The idea was that this would be akin to a system message used for OpenAI's models. As a result, in the event that the conversation does not fit in the configured context window, it will remove the *second* message in the conversation until the conversation fits. If, for whatever reason, you do not want this behavior, and want to discard the first message so that the model only remembers the most recent messages, change the `1` in `messages.pop(1)` to `0`.

Does not support GPT-Q or other quantized models out-of-the-box because I don't use them because they're slow in my experience. To enable support, set up the python virtual environment and install `optimum` and `auto-gptq` with `pip install auto-gptq optimum`. Then update the `requirements.txt` with `pip freeze > requirements.txt`. From there, running the build script should result in a docker image that allows you to use them from huggingface, but YMMV since I haven't tested it as of writing this.