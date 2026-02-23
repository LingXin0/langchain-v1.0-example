import asyncio
import time
from core.llm import get_model_anthropic


def test_sync_batch():
    """
    Example of synchronous batch call using LangChain's .batch() method.
    """
    print("\n--- Starting Synchronous Batch ---")
    model = get_model_anthropic()

    prompts = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
    ]

    start_time = time.time()
    # Synchronous batch call - requests are executed in parallel by default in LangChain batch
    results = model.batch(prompts)
    end_time = time.time()

    for prompt, result in zip(prompts, results):
        print(f"Prompt: {prompt}")
        print(f"Response: {result.content.strip()}")
        print("-" * 20)

    print(f"Synchronous batch completed in {end_time - start_time:.2f} seconds")


async def test_async_batch():
    """
    Example of asynchronous batch call using LangChain's .abatch() method.
    """
    print("\n--- Starting Asynchronous Batch ---")
    model = get_model_anthropic()

    prompts = [
        "Tell me a short joke about Python.",
        "Tell me a short joke about Java.",
        "Tell me a short joke about C++.",
    ]

    start_time = time.time()
    # Asynchronous batch call - allows for better concurrency in async environments
    results = await model.abatch(prompts)
    end_time = time.time()

    for prompt, result in zip(prompts, results):
        print(f"Prompt: {prompt}")
        print(f"Response: {result.content.strip()}")
        print("-" * 20)

    print(f"Asynchronous batch completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # Run synchronous test
    test_sync_batch()

    # Run asynchronous test
    asyncio.run(test_async_batch())
