# client.py - Client for interacting with the Llama 3.2 API server
import argparse
import requests


def main():
    parser = argparse.ArgumentParser(description="Client for Llama 3.2 API Server")
    parser.add_argument(
        "--host", type=str, default="http://localhost:8000", help="API server host"
    )
    parser.add_argument("--question", type=str, required=True, help="Question to ask")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for generation"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Maximum tokens to generate"
    )

    args = parser.parse_args()

    # Build the request
    request_data = {
        "question": args.question,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    try:
        # Send the request to the API server
        response = requests.post(f"{args.host}/predict", json=request_data)

        # Check if the request was successful
        response.raise_for_status()

        # Parse the response
        result = response.json()

        # Print the response
        print("\nQuestion:")
        print(args.question)

        if result.get("reasoning"):
            print("\nReasoning:")
            print(result["reasoning"])

        print("\nAnswer:")
        print(result["answer"])

    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        if hasattr(e, "response") and e.response:
            print(f"Status code: {e.response.status_code}")
        try:
            error_detail = e.response.json()
            print(f"Error detail: {error_detail}")
        except ValueError:
            print(f"Error content: {e.response.text}")


if __name__ == "__main__":
    main()
