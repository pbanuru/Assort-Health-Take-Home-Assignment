# Assort Phone Agent

## Overview
Assort Phone Agent is an appointment scheduling voice agent built using LiveKit and Twilio.

## Prerequisites
- Python 3.10.12
- Dependencies as listed in:
[LiveKit Voice Agent Quickstart Guide](https://docs.livekit.io/agents/quickstarts/voice-agent/) for the voice agent
and
[LiveKit Inbound Calls Quickstart Guide](https://docs.livekit.io/agents/quickstarts/inbound-calls/) for the phone call handling (This project uses Twilio)

## Setup Instructions

1. Clone this repository:
   ```
   git clone https://github.com/your-username/Assort-Health-Take-Home-Assignment.git
   cd Assort-Health-Take-Home-Assignment/assort-phone-agent
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the necessary environment variables:

   ```bash
   # LLM and TTS
   export OPENAI_API_KEY=<your-openai-api-key>
   
   # Speech-to-Text
   export DEEPGRAM_API_KEY=<your-deepgram-api-key>
   
   # Phone Call Integration
   export TWILIO_ACCOUNT_SID=<your-twilio-account-sid>
   export TWILIO_AUTH_TOKEN=<your-twilio-auth-token>
   export TWILIO_PHONE_NUMBER=<your-twilio-phone-number>
   
   # Agent & Phone Call Integration
   export LIVEKIT_SIP_URI=<your-livekit-sip-uri>
   export LIVEKIT_URL=<your-livekit-url>
   export LIVEKIT_API_KEY=<your-livekit-api-key>
   export LIVEKIT_API_SECRET=<your-livekit-api-secret>
   
   # Email Integration
   export GMAIL_USER=<your-gmail-user>
   export GMAIL_PASSWORD=<your-gmail-password>
   ```

   Replace `<your-*>` placeholders with your actual API keys and credentials.

## Running the Application

To start the Assort Phone Agent, run:
```
python agent.py dev
```