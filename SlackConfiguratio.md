# Slack Bot Configuration (Blueprint)

## Overview
This document describes how to configure and integrate a Slack bot using **Slack Bolt (Python)** and **Flask**.

The bot listens to Slack events, processes incoming messages, and responds when mentioned.

---

## Slack App Setup

### 1. Create Slack App
1. Go to Slack API dashboard  
2. Click **Create New App**  
3. Choose **From scratch**  
4. Select your workspace  

---

### 2. OAuth & Permissions

Add the following **Bot Token Scopes**:

- `app_mentions:read` – detect when bot is mentioned  
- `channels:history` – read messages from channels  
- `chat:write` – send messages  
- `files:read` – access uploaded files  

After that:
- Install the app to your workspace  
- Copy **Bot User OAuth Token**

---

### 3. Enable Event Subscriptions

1. Turn on **Enable Events**  
2. Set Request URL: https://your-domain.com/slack/events

3. Subscribe to bot events:

- `app_mention`  
- `message.channels`  

---

## Flask Integration

Slack events are received via a Flask endpoint and handled using **SlackRequestHandler**.

### Initialization

```python
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask

app = App(token="SLACK_BOT_TOKEN")
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)
```

## Slack Events Endpoint
```python
from flask import request, jsonify

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json

    # Slack URL verification
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    return handler.handle(request)
```

## Event Handling
1. Handling All Messages
   ```python
   @app.event("message")
   def handle_all_messages(body):
    text = body["event"]["text"]
   ```
   * Triggered for every message in channels
   * Can be used for logging or preprocessing
   * Typically ignores bot mentions
2. Handling Bot Mentions
   ```python
   @app.event("app_mention")
   def handle_mentions(body, say):
    text = body["event"]["text"]
    say(text="Response message")
   ```
   * Triggered when bot is mentioned (@bot)
   * Extracts message text
   * Sends response using say()
