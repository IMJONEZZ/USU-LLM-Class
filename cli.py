#!/usr/bin/env python3
import argparse
import os
import sys
import time
from prompt_optimizer import PromptOptimizer


def get_api_key():
    """Get the OpenAI API key from environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OpenAI API key not found! Please set the OPENAI_API_KEY environment variable."
        )
        sys.exit(1)
    return api_key


def stream_response(text, delay=0.01):
    """Stream the response to the terminal with a typing effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def chat_mode():
    """Interactive chat mode for conversation with the LLM."""
    api_key = get_api_key()
    optimizer = PromptOptimizer("gpt-3.5-turbo", api_key=api_key)

    print("Chat mode initialized. Type 'exit' to quit.")
    print("Enter your prompt: ")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        if not user_input:
            continue

        # Simple streaming implementation for text response
        print("\nGenerating response...")
        response = optimizer.generate_text(user_input)
        print("\n")
        stream_response(response)
        print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Command Line Interface for PromptOptimizer"
    )

    # Add subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run")

    # Chat mode
    subparsers.add_parser("chat", help="Interactive chat mode")

    # HTML generation mode
    html_parser = subparsers.add_parser("html", help="Generate HTML content")
    html_parser.add_argument("topic", help="Topic for HTML generation")
    html_parser.add_argument(
        "--type",
        choices=["article", "product"],
        default="article",
        help="Content type (article or product)",
    )
    html_parser.add_argument("--output", help="Output file to save HTML (optional)")

    # Structured output mode
    structured_parser = subparsers.add_parser(
        "structured", help="Generate structured output"
    )
    structured_parser.add_argument("topic", help="Topic for structured output")
    structured_parser.add_argument(
        "--format",
        choices=["json", "xml"],
        default="json",
        help="Output format (json or xml)",
    )
    structured_parser.add_argument(
        "--output", help="Output file to save structured data (optional)"
    )

    args = parser.parse_args()

    # If no arguments are provided, show help and exit
    if not args.mode:
        parser.print_help()
        sys.exit(0)

    # Get API key from environment
    api_key = get_api_key()

    # Initialize the optimizer
    optimizer = PromptOptimizer("gpt-3.5-turbo", api_key=api_key)

    # Process based on mode
    if args.mode == "chat":
        chat_mode()

    elif args.mode == "html":
        print(f"Generating HTML for topic: {args.topic}")
        html_content = optimizer.generate_html(args.topic, args.type)

        if args.output:
            with open(args.output, "w") as f:
                f.write(html_content)
            print(f"HTML content saved to {args.output}")
        else:
            stream_response(html_content, delay=0.001)

    elif args.mode == "structured":
        print(f"Generating structured output for topic: {args.topic}")
        output = optimizer.generate_structured_output(args.topic, args.format)

        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Structured output saved to {args.output}")
        else:
            stream_response(output, delay=0.001)


if __name__ == "__main__":
    main()
