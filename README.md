# Nuclear Geek's Flux Capacitor

A project utilizing [Black Forest Lab's](https://blackforestlabs.ai/) FLUX model for generating images via a web application.

## Prerequisites

1. **Hugging Face Access Token**:
   - You need to request access to the FLUX model: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.1-dev)
   - Once you have access, obtain your Hugging Face token.

2. **Environment Setup**:
   - Add your Hugging Face token to an `.env` file at the root of the project. Your `.env` file should look like this:
     ```
     HF_TOKEN=your_token_here
     ```

## Getting Started

Follow these steps to set up the environment and run the application:

1. **Run the setup script**:
   - Execute `start_flux.bat` to install the necessary environment and dependencies, and to run the script.
   ```plaintext
   ./start_flux.bat