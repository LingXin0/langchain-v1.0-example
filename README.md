# langchain-v1.0-example
Demo for langchain structure

## Installation

Follow these steps to set up the project locally:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd langchain-v1.0-example
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**

    *   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .venv\Scripts\activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up environment variables:**
    Copy the example environment file and update it with your API keys:
    ```bash
    cp .env.example .env
    ```
    Edit `.env` and add your keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

## Dependency Management

To export your current environment's dependencies to `requirements.txt`:

```bash
pip freeze > requirements.txt
```
